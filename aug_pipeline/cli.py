from __future__ import annotations

import argparse
import json
import time

from .config import ARMS, STAGES, Config


def main():
    parser = argparse.ArgumentParser(description='TCP4D DROID augmentation workbench')
    parser.add_argument('--config', default=None, help='YAML config; paths inside resolve from the TCP4D project root')
    sub = parser.add_subparsers(dest='command', required=True)
    web = sub.add_parser('web')
    web.add_argument('--host', default='127.0.0.1')
    web.add_argument('--port', type=int, default=7862)
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--gpu', action='store_true', help='Also initialize CUDA, cuRobo and SAPIEN')
    download = sub.add_parser('download')
    download.add_argument('components', nargs='+', choices=['all', 'tcp4d', 'assets', 'sam3', 'propainter'])
    download.add_argument('--arms', nargs='+', choices=ARMS, default=['ur5_wsg', 'franka_panda'])
    run = sub.add_parser('run')
    run.add_argument('--episode')
    run.add_argument('--camera')
    run.add_argument('--fps', type=float, help='RGB sequence frame rate (default: metadata or 15); videos use their embedded rate')
    run.add_argument('--arms', nargs='+', choices=ARMS, default=['ur5_wsg', 'franka_panda'])
    run.add_argument('--mask-method', choices=['sam3'], default=None)
    run.add_argument('--query', nargs=2, type=float)
    run.add_argument('--text', default='robot arm')
    run.add_argument('--job', help='Resume an existing job ID')
    run.add_argument('--stage', choices=STAGES, default='track')
    run.add_argument('--force', action='store_true')
    args = parser.parse_args()
    config = Config(args.config)
    if args.command == 'web':
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(config), host=args.host, port=args.port)
    elif args.command == 'doctor':
        from .resources import doctor
        results = doctor(config, args.gpu)
        print(json.dumps(results, indent=2))
        # Uninstalled optional robot assets do not make the base stack unusable.
        optional = set()
        if config['mask_method'] != 'sam3':
            optional.update(['sam3', 'sam3_checkpoint'])
        required = [r for r in results if not r['name'].startswith('asset:') and r['name'] not in optional]
        if not all(r['ok'] for r in required):
            raise SystemExit(1)
    elif args.command == 'download':
        from .resources import download
        download(config, args.components, args.arms)
    else:
        from .jobs import JobManager
        manager = JobManager(config)
        try:
            if args.job:
                job = manager.get(args.job)
            else:
                if not args.episode:
                    parser.error('run needs --episode or --job')
                request = dict(episode=args.episode, camera=args.camera, arms=args.arms,
                               query=args.query, text=args.text, fps=args.fps)
                if args.mask_method:
                    request['mask_method'] = args.mask_method
                job = manager.create(request)
            print('Job:', job['id'], flush=True)
            manager.start(job['id'], args.stage, args.force)
            log = manager.directory(job['id']) / 'run.log'
            cursor = 0
            while True:
                if log.exists():
                    with log.open() as f:
                        f.seek(cursor)
                        print(f.read(), end='', flush=True)
                        cursor = f.tell()
                job = manager.get(job['id'])
                if job['status'] not in ('queued', 'running'):
                    print(json.dumps(job, indent=2))
                    if job['status'] in ('failed', 'cancelled', 'interrupted'):
                        raise SystemExit(1)
                    if job['status'] == 'partial':
                        raise SystemExit(2)
                    break
                time.sleep(.5)
        except KeyboardInterrupt:
            manager.cancel(job['id'])
            raise SystemExit(130)
        finally:
            manager.close()


if __name__ == '__main__':
    main()
