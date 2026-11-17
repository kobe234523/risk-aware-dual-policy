import contextlib
import csv
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))
from risk_learning.reference import ReferenceConfig, continuous_reference, reference_from_env
from risk_learning.model import RiskHead, observable_context, encoder_signature, load_head, predict_risk, FEATURE_SPEC, FORMAT_VERSION
from risk_learning.tcp_adapter import tcp_actions_and_features
from risk_learning.data import save_episode, inspect_shards, split_groups, FeatureDataset
from risk_learning.runtime import compute_online_risk, validate_options
import train_risk_head


class TinyPerception(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, 1)

    def forward(self, image):
        spatial = self.conv(torch.nn.functional.adaptive_avg_pool2d(image,(4,5)))
        return spatial.mean((2, 3)), spatial


class TinyTCP(nn.Module):
    def __init__(self):
        super().__init__()
        self.perception = TinyPerception()
        self.calls = 0

    def forward(self, image, state, target):
        self.calls += 1
        self.last_state = state.detach().clone()
        self.perception(image)
        return {"pred_wp": torch.zeros(1, 4, 2)}

    def process_action(self, *args):
        return .1, .2, 0, {}

    def control_pid(self, *args):
        return -.1, 0, .3, {}


def checkpoint(path, net, head=None):
    torch.save(dict(format_version=FORMAT_VERSION, feature_spec=FEATURE_SPEC,
                    encoder_signature=encoder_signature(net), state_dict=(head or RiskHead()).state_dict(),
                    reference_config=asdict(ReferenceConfig())), str(path))


class ReferenceTests(unittest.TestCase):
    def test_exact_fusion_and_multiple_pedestrians(self):
        expected = math.exp(-1.5**2 / 18) * (.5 * .5 + .5 * .5)
        value = continuous_reference([0, 0], [2, 0], 5, [[7.5, 1.5]])
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(continuous_reference([0, 0], [1, 0], 5, [[7.5, 1.5], [30, 20]]), value)

    def test_empty_behind_low_speed_and_invalid_state(self):
        self.assertEqual(continuous_reference([0, 0], [1, 0], 0, []), 0)
        self.assertEqual(continuous_reference([0, 0], [1, 0], 10, [[-1, 0], [0, 0]]), 0)
        self.assertTrue(np.isfinite(continuous_reference([0, 0], [1, 0], 0, [[.1, 0]])))
        for position in ([np.nan, 0], [np.inf, 0]):
            with self.assertRaises(ValueError):
                continuous_reference([0, 0], [1, 0], 1, [position])
        with self.assertRaises(ValueError):
            continuous_reference([0, 0], [0, 0], 1, [])

    def test_all_walkers_are_used_not_only_probe(self):
        class Actor:
            is_alive = True
            def __init__(self, identity, x): self.id, self.x = identity, x
            def get_location(self): return types.SimpleNamespace(x=self.x, y=0)
        far, near = Actor(1, 100), Actor(2, 2)
        vehicle = types.SimpleNamespace(get_location=lambda: types.SimpleNamespace(x=0,y=0),
                   get_transform=lambda: types.SimpleNamespace(get_forward_vector=lambda: types.SimpleNamespace(x=1,y=0)),
                   get_velocity=lambda: types.SimpleNamespace(x=5,y=0))
        env = types.SimpleNamespace(vehicle=vehicle, walker_probe=far, walker_probes=[far, near])
        self.assertGreater(reference_from_env(env), .5)


class ModelTests(unittest.TestCase):
    def test_context_dimensions_and_normalization(self):
        ctx = observable_context(15, [60, -15], 4)
        np.testing.assert_allclose(ctx[:3], [1.5, 1, -.5])
        self.assertEqual(ctx.shape, (9,))
        np.testing.assert_array_equal(ctx[3:], [0, 0, 0, 1, 0, 0])
        with self.assertRaises(ValueError): observable_context(1, [0, 1], 7)

    def test_architecture_checkpoint_and_signature_guard(self):
        head = RiskHead()
        self.assertEqual(sum(p.numel() for p in head.parameters()), 166657)
        net = TinyTCP().eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'head.pt'
            checkpoint(path, net, head)
            loaded, _ = load_head(path, encoder_signature(net), torch.device('cpu'))
            self.assertTrue(0 <= predict_risk(loaded, np.zeros(521, np.float32)) <= 1)
            with self.assertRaises(ValueError): load_head(path, 'other-encoder', torch.device('cpu'))
            with self.assertRaises(ValueError): predict_risk(loaded, np.full(521, np.nan, np.float32))

    def test_single_forward_features_and_original_tcp_scaling(self):
        net = TinyTCP().eval()
        traj, ctrl, features = tcp_actions_and_features(net, np.zeros((256, 900, 3),np.uint8), 12, [0,20], 4)
        self.assertEqual(net.calls, 1)
        self.assertEqual(features.shape, (521,))
        self.assertAlmostEqual(float(net.last_state[0,0]), 1.0)
        self.assertAlmostEqual(float(features[512]), 1.2, places=5)
        np.testing.assert_allclose(traj, [-.1,-.3])
        np.testing.assert_allclose(ctrl, [.1,.2])
        self.assertFalse(net.perception._forward_hooks)

    def test_learned_inference_never_calls_privileged_callback(self):
        def forbidden(): raise AssertionError('privileged risk entered learned inference')
        result = compute_online_risk('learned', RiskHead().eval(), np.zeros(521,np.float32), legacy=forbidden)
        self.assertFalse(result['uses_privileged_risk'])
        with self.assertRaises(ValueError): compute_online_risk('learned', legacy=forbidden)
        with self.assertRaises(ValueError):
            validate_options(types.SimpleNamespace(delta=.5,method='ours',risk_estimator='learned',risk_head_ckpt='',collect_risk_data=False))


class DataTrainingTests(unittest.TestCase):
    def create_shards(self, root, net):
        metadata = dict(scene_group='', encoder_signature=encoder_signature(net), reference_config=asdict(ReferenceConfig()))
        for group in range(5):
            for weather in ['sun','rain']:
                meta = dict(metadata,scene_group='layout_{}'.format(group),weather=weather)
                x = np.random.RandomState(group).randn(4,521).astype(np.float32)
                save_episode(root/'{}_{}.npz'.format(group,weather),x,[0,.2,.6,1],[0,1,2,3],meta)

    def test_group_split_keeps_weather_variants_together(self):
        with tempfile.TemporaryDirectory() as temp:
            self.create_shards(Path(temp),TinyTCP())
            records = inspect_shards(temp)
            split = split_groups(records)
            owner={}
            for partition,rows in split.items():
                self.assertGreater(len(FeatureDataset(rows)),0)
                for r in rows:
                    group=r['metadata']['scene_group']
                    self.assertEqual(owner.setdefault(group,partition),partition)
            self.assertEqual(split_groups(records),split)

    def test_mixed_encoder_and_too_few_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            meta=dict(scene_group='one',encoder_signature='A',reference_config=asdict(ReferenceConfig()))
            save_episode(root/'a.npz',[np.zeros(521)],[0],[0],meta)
            with self.assertRaises(ValueError): split_groups(inspect_shards(root))
            save_episode(root/'b.npz',[np.zeros(521)],[0],[0],dict(meta,encoder_signature='B'))
            with self.assertRaises(ValueError): inspect_shards(root)

    def test_offline_training_and_independent_test_round_trip(self):
        # Small temporary tensors test the pipeline, not driving performance.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.create_shards(root/'data',TinyTCP())
            args=train_risk_head.parse_args(['train','--data_dir',str(root/'data'),'--output_dir',str(root/'fit'),'--epochs','2','--batch_size','8'])
            train_risk_head.train(args)
            self.assertTrue((root/'fit/risk_head.pt').exists())
            self.assertFalse((root/'fit/test_metrics.json').exists())
            args=train_risk_head.parse_args(['test','--checkpoint',str(root/'fit/risk_head.pt'),'--manifest',str(root/'fit/split_manifest.json'),'--output',str(root/'test.json')])
            train_risk_head.test(args)
            result=json.loads((root/'test.json').read_text())
            self.assertTrue(math.isfinite(result['mse']))


class EvaluationIntegrationTests(unittest.TestCase):
    def test_one_step_learned_rollout_without_privileged_risk(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); net=TinyTCP().eval(); head_path=root/'head.pt'
            forced_head=RiskHead()
            with torch.no_grad():
                for parameter in forced_head.parameters():parameter.zero_()
                forced_head.layers[4].bias.fill_(2.)
            checkpoint(head_path,net,forced_head)
            class Env:
                def __init__(self):
                    self.t=10.;self.finished=False
                    self.observation_space=types.SimpleNamespace(shape=(3,4,5))
                    self.action_space=types.SimpleNamespace(shape=(2,))
                    self.frame = 100
                    self.world = types.SimpleNamespace(get_snapshot=self.get_snapshot)

                def get_snapshot(self):
                    return types.SimpleNamespace(
                        frame=self.frame,
                        timestamp=types.SimpleNamespace(elapsed_seconds=self.t),
                    )
                def reset(self): return np.zeros((3,4,5),np.float32)
                def get_tcp_inputs(self): return np.zeros((256,900,3),np.uint8),3.,[0,20],4
                def step(self,a):
                    self.t+=.15
                    self.frame += 1
                    return None,1.,True,dict(speed=2.,success=True,steer=.1,throttle=.2,brake=0)
                def finish(self):self.finished=True
            env=Env()
            agent=types.SimpleNamespace(load_best=lambda _:None,select_action=lambda _:np.array([0,-.5],np.float32))
            class Video:
                def __init__(self,*a):pass
                def init(self,**k):pass
                def record(self,*a):pass
                def save(self,*a):pass
            utils=types.SimpleNamespace(set_seed_everywhere=lambda _:None,make_dir=lambda p:(Path(p).mkdir(parents=True,exist_ok=True) or p),eval_mode=lambda _:contextlib.nullcontext(),FrameStack=lambda env,**_:env)
            mocks={
                'utils':types.SimpleNamespace(utils=utils), 'utils.video':types.SimpleNamespace(VideoRecorder=Video),
                'carla_env.carla_env':types.SimpleNamespace(CarlaEnv=lambda **_:env),
                'train_carla_off_policy':types.SimpleNamespace(make_agent=lambda **_:agent),
                'train_carla_on_policy':types.SimpleNamespace(get_tcp_actions=lambda *a:None,load_tcp=lambda *a:(net,None)),
            }
            with patch.dict(sys.modules,mocks),patch.object(torch.cuda,'is_available',return_value=False):
                spec=importlib.util.spec_from_file_location('staged_evaluator',str(RL/'eval_ghost_dual_policy.py'))
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                def forbidden(*args,**kwargs):raise AssertionError('Unexpected privileged risk access')
                module.compute_risk=forbidden;module.reference_from_env=forbidden
                with patch.object(sys,'argv',['eval','--method','ours','--risk_head_ckpt',str(head_path),'--work_dir',str(root/'eval'),'--rl_model_dir','unused','--num_eval_episodes','1']):
                    module.evaluate(module.parse_args())
                module.reference_from_env=lambda *args:.7
                with patch.object(sys,'argv',['eval','--method','tcp','--collect_risk_data','--risk_scene_group','layout_A','--work_dir',str(root/'collect'),'--num_eval_episodes','1']):
                    module.evaluate(module.parse_args())
                shard=next((root/'collect').rglob('*.npz'))
                with np.load(str(shard),allow_pickle=False) as data:
                    self.assertEqual(data['features'].shape,(1,521))
                    self.assertAlmostEqual(float(data['labels'][0]),.7)
                    self.assertEqual(json.loads(str(data['metadata'].item()))['scene_group'],'layout_A')
            self.assertTrue(env.finished)
            log=next((root/'eval').rglob('frames.csv'))
            with log.open() as stream:row=next(csv.DictReader(stream))
            self.assertEqual(row['risk_backend'],'learned')
            self.assertEqual(int(row['input_world_frame']), 100)
            self.assertEqual(int(row['next_world_frame']), 101)
            self.assertEqual(row['uses_privileged_risk'],'0')
            self.assertEqual(row['selected_policy'],'rl')
            self.assertEqual(row['takeover'],'1')
            self.assertAlmostEqual(float(row['final_brake']),.5)
            self.assertEqual(row['risk_reference'],'')
            self.assertEqual(float(row['input_speed']),3.)
            self.assertEqual(float(row['next_speed']),2.)
            self.assertAlmostEqual(float(row['next_sim_time']),.15)


if __name__=='__main__':unittest.main()
