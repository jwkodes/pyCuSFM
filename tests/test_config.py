# SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from pycusfm import CuSFMConfig, create_cusfm_runner
from pycusfm.config import _native_scalar, prepare_native_config
from pycusfm.command_runner import CommandRunner

PACKAGE = Path(__file__).resolve().parents[1] / 'pycusfm'


class ConfigTests(unittest.TestCase):

    def test_validation_and_unknown_yaml(self):
        for kwargs in ({'matcher_processes':
                        0}, {'matcher_num_threads':
                             True}, {'search_translation_threshold_meters':
                                     float('nan')}, {'matcher_gpu_ids':
                                                     [0, 0]}, {'skip_cuvslam':
                                                               'false'},
                       {'search_rotation_threshold_degrees':
                        181}, {'matcher_gpu_ids': 0}, {'matcher_gpu_ids':
                                                       [[0]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CuSFMConfig(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.yaml'
            p.write_text('skip_cuvslam: true\nmatcher_gpu_ids: [0, 1]\n')
            config = CuSFMConfig.from_file(p)
            self.assertTrue(config.skip_cuvslam)
            self.assertEqual(config.matcher_gpu_ids, (0, 1))
            p.write_text('unknown_setting: 1\n')
            with self.assertRaises(ValueError):
                CuSFMConfig.from_file(p)

    def test_av_defaults_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = create_cusfm_runner(
                package_path=PACKAGE, config_set='av', cusfm_base_dir=tmp)
            runner._prepare_configuration()
            values = runner.effective_config['configuration']
            self.assertEqual(values['search_method'], 'TRACK_2D_OVERLAP_IR')
            self.assertEqual(values['mapping_num_threads'], 12)
            self.assertEqual(values['ba_num_threads'], 12)
            self.assertTrue(values['skip_cuvslam'])
            self.assertEqual(values['min_inter_frame_distance'], 0.)

    def test_scoped_native_fields_and_comments(self):
        text = '# num_threads: 99\nnum_threads: 8\nbundle_adjustment_config { num_threads: 3 }\n'
        result = _native_scalar(
            text, ('bundle_adjustment_config', 'num_threads'), 5)
        self.assertEqual(_native_scalar(result, ('num_threads', )), 8)
        self.assertEqual(
            _native_scalar(
                result, ('bundle_adjustment_config', 'num_threads')), 5)
        self.assertTrue(result.startswith('# num_threads: 99'))
        with self.assertRaises(ValueError):
            _native_scalar(
                'num_threads: 1\nnum_threads: 2', ('num_threads', ))

    def test_native_snapshot_preserves_sources_and_defaults(self):
        source = PACKAGE / 'configs/isaac'
        before = {
            p.name: p.read_bytes()
            for p in source.iterdir() if p.is_file()
        }
        with tempfile.TemporaryDirectory() as tmp:
            target, effective, hashes = prepare_native_config(
                source, tmp, CuSFMConfig())
            self.assertEqual(
                effective['search_translation_threshold_meters'], 3)
            self.assertEqual(
                effective['search_method'], 'POSE_GRAPH_ASSOCIATION')
            self.assertEqual(effective['mapping_num_threads'], 8)
            config = CuSFMConfig(
                search_method='RADIUS_SEARCH',
                search_translation_threshold_meters=0.05,
                mapping_num_threads=2,
                ba_num_threads=3,
                max_frame_pair_size_per_file=128)
            changed, values, _ = prepare_native_config(source, tmp, config)
            self.assertNotEqual(target, changed)
            self.assertEqual(values['search_method'], 'RADIUS_SEARCH')
            self.assertEqual(values['mapping_num_threads'], 2)
            self.assertEqual(values['ba_num_threads'], 3)
            self.assertEqual(values['max_frame_pair_size_per_file'], 128)
            self.assertEqual(
                hashes,
                prepare_native_config(source, tmp, CuSFMConfig())[2])
        self.assertEqual(
            before,
            {p.name: p.read_bytes()
             for p in source.iterdir() if p.is_file()})

    def test_factory_cli_precedence_and_effective_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = CuSFMConfig(
                skip_cuvslam=True,
                min_inter_frame_distance=.025,
                matcher_num_threads=2,
                matcher_processes=2,
                search_translation_threshold_meters=.05)
            runner = create_cusfm_runner(
                package_path=PACKAGE,
                config_set='isaac',
                config=config,
                cusfm_base_dir=tmp,
                min_inter_frame_distance=.03,
                matcher_num_threads=3)
            runner._prepare_configuration()
            saved = json.loads(
                (Path(tmp) / 'effective_config.json').read_text())
            self.assertTrue(saved['configuration']['skip_cuvslam'])
            self.assertEqual(
                saved['configuration']['min_inter_frame_distance'], .03)
            self.assertEqual(saved['configuration']['matcher_num_threads'], 3)
            self.assertEqual(saved['configuration']['matcher_processes'], 2)
            self.assertEqual(
                saved['configuration']['search_translation_threshold_meters'],
                .05)
            self.assertEqual(saved['configuration']['mapping_num_threads'], 8)
            self.assertTrue(Path(saved['native_config_directory']).is_dir())
            self.assertEqual(config.matcher_num_threads, 2)

    def test_cli_file_and_explicit_flag_precedence(self):
        from pycusfm import cusfm_cli
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pipeline.yaml'
            path.write_text(
                'skip_cuvslam: true\nmatcher_processes: 2\n'
                'min_inter_frame_distance: 0.025\n')
            argv = [
                'cusfm_cli', '--pipeline_config',
                str(path), '--cusfm_base_dir',
                str(Path(tmp) / 'out'), '--input_dir', 'input',
                '--min_inter_frame_distance', '0.03', '--matcher_processes',
                '3', '--skip_cuvslam', 'false'
            ]
            # Capture the real constructed runner to inspect resolved values.
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(cusfm_cli, 'create_cusfm_runner', wraps=create_cusfm_runner) as factory, \
                    mock.patch('pycusfm.cusfm_runner.CuSFMRunner.run_all'), \
                    mock.patch.object(cusfm_cli, 'print_params'):
                cusfm_cli.main()
                config = factory.call_args.kwargs['config']
            self.assertFalse(config.skip_cuvslam)
            self.assertEqual(config.min_inter_frame_distance, .03)
            self.assertEqual(config.matcher_processes, 3)
            self.assertFalse((Path(tmp) / 'out').exists())

    def test_cli_validates_extrinsics_from_file(self):
        from pycusfm import cusfm_cli
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pipeline.yaml'
            path.write_text('optimize_extrinsics: true\n')
            argv = [
                'cusfm_cli', '--pipeline_config',
                str(path), '--cusfm_base_dir',
                str(Path(tmp) / 'out'), '--input_dir', 'input'
            ]
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch('pycusfm.cusfm_runner.CuSFMRunner.run_all') as run:
                with self.assertRaisesRegex(ValueError, 'vehicle_rig'):
                    cusfm_cli.main()
                run.assert_not_called()

    def test_skip_cuvslam_uses_supplied_input_not_stale_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = create_cusfm_runner(
                package_path=PACKAGE,
                config_set='isaac',
                config=CuSFMConfig(
                    skip_cuvslam=True,
                    override_frames_meta_file=str(root / 'external.json')),
                input_dir=str(root / 'input'),
                cusfm_base_dir=str(root / 'out'))
            stale = Path(
                runner.cuvslam_output_dir) / 'frames_meta_cuvslam.json'
            stale.parent.mkdir(parents=True)
            stale.write_text('{}')
            with mock.patch.object(runner.runner, 'run_binary') as binary:
                self.assertEqual(
                    runner.run_cuvslam(
                        runner.input_dir, runner.cuvslam_output_dir,
                        runner.override_frames_meta_file),
                    str(root / 'external.json'))
                binary.assert_not_called()

    def test_direct_matcher_passes_threads_processes_and_gpu_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = create_cusfm_runner(
                package_path=PACKAGE,
                config_set='isaac',
                cusfm_base_dir=tmp,
                config=CuSFMConfig(
                    matcher_num_threads=3,
                    matcher_processes=2,
                    matcher_gpu_ids=(0, 1)))
            tasks = Path(runner.tasks_dir)
            tasks.mkdir(parents=True)
            (tasks / '0.pb').write_bytes(b'')
            with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
                    mock.patch.object(runner.runner, 'remove_directory'), \
                    mock.patch.object(runner.runner, 'run_binary'), \
                    mock.patch.object(runner.runner, 'run_binaries_parallel', return_value=True) as parallel:
                runner.match_features()
            kwargs = parallel.call_args.kwargs
            self.assertEqual(kwargs, dict(max_workers=2, gpu_ids=(0, 1)))
            args = parallel.call_args.args[0][0]['args']
            self.assertEqual(args[args.index('--num_thread') + 1], '3')

    def test_matcher_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = create_cusfm_runner(
                package_path=PACKAGE, config_set='isaac', cusfm_base_dir=tmp)
            Path(runner.tasks_dir).mkdir(parents=True)
            (Path(runner.tasks_dir) / '0.pb').write_bytes(b'')
            with mock.patch.object(runner.runner, 'remove_directory'), \
                    mock.patch.object(runner.runner, 'run_binary'), \
                    mock.patch.object(runner.runner, 'run_binaries_parallel', return_value=False):
                with self.assertRaises(RuntimeError):
                    runner.match_features()


class GpuSchedulingTests(unittest.TestCase):

    def test_visible_device_mapping(self):
        with mock.patch.dict(os.environ,
                             {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'}):
            self.assertEqual(
                CommandRunner.resolve_gpu_ids((1, 0)), ['GPU-b', 'GPU-a'])
            with self.assertRaises(ValueError):
                CommandRunner.resolve_gpu_ids((2, ))
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(ValueError):
                CommandRunner.resolve_gpu_ids((0, ))

    def test_parallel_gpu_slots_do_not_mutate_parent_environment(self):
        runner = CommandRunner('/unused', '/unused')
        active, observed = {}, []
        lock = threading.Lock()

        def run(*args, **kwargs):
            device = kwargs['env_overrides']['CUDA_VISIBLE_DEVICES']
            with lock:
                active[device] = active.get(device, 0) + 1
                self.assertEqual(active[device], 1)
                observed.append(device)
            time.sleep(.02)
            with lock:
                active[device] -= 1
            return 0

        commands = [
            dict(binary_name='matcher', args=[str(i)], log_file=None)
            for i in range(6)
        ]
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}), \
                mock.patch.object(runner, 'run_binary', side_effect=run):
            self.assertTrue(
                runner.run_binaries_parallel(
                    commands, max_workers=2, gpu_ids=(0, 1)))
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '3,5')
        self.assertEqual(set(observed), {'3', '5'})

    def test_dry_run_does_not_need_tensorrt_or_spawn(self):
        runner = CommandRunner('/unused', '/unused', dry_run=True)
        with mock.patch('subprocess.Popen') as spawn:
            self.assertEqual(runner.run_binary('matcher', []), 0)
            spawn.assert_not_called()

    def test_command_arguments_are_not_shell_expanded(self):
        runner = CommandRunner('/unused', '/unused', add_tensorrt_path=False)
        process = mock.Mock(returncode=0)
        process.stdout.readline.return_value = ''
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                'subprocess.Popen', return_value=process) as spawn:
            runner.run_command(
                '/some path/tool', ['literal $(test)'],
                str(Path(tmp) / 'log.txt'))
        self.assertEqual(
            spawn.call_args.args[0], ['/some path/tool', 'literal $(test)'])
        self.assertNotIn('shell', spawn.call_args.kwargs)
