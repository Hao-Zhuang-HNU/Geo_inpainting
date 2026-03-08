import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Remove paths listed in rm_list file")
    parser.add_argument('--rm_list', type=str, required=True, help='Path to list file containing paths to remove')
    return parser.parse_args()


def load_paths(rm_list_path):
    paths = []
    with open(rm_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            path = line.split('|')[0].strip()
            if path:
                paths.append(path)
    return paths


def remove_path(path):
    norm_path = os.path.normpath(path)

    if not os.path.exists(norm_path):
        print(f"[SKIP] Not found: {norm_path}")
        return 'skip'

    try:
        if os.path.isdir(norm_path) and not os.path.islink(norm_path):
            shutil.rmtree(norm_path)
            print(f"[OK] Removed directory: {norm_path}")
        else:
            os.remove(norm_path)
            print(f"[OK] Removed file/link: {norm_path}")
        return 'ok'
    except Exception as e:
        print(f"[ERR] Failed to remove {norm_path}: {e}")
        return 'err'


def main():
    args = parse_args()

    if not os.path.exists(args.rm_list):
        print(f"Error: rm_list file not found: {args.rm_list}")
        return

    paths = load_paths(args.rm_list)
    print(f"Loaded {len(paths)} paths from {args.rm_list}")

    count_ok = 0
    count_skip = 0
    count_err = 0

    for path in paths:
        result = remove_path(path)
        if result == 'ok':
            count_ok += 1
        elif result == 'skip':
            count_skip += 1
        else:
            count_err += 1

    print("\n" + "=" * 40)
    print("REMOVE SUMMARY")
    print("=" * 40)
    print(f"Total listed: {len(paths)}")
    print(f"Removed:      {count_ok}")
    print(f"Skipped:      {count_skip}")
    print(f"Errors:       {count_err}")


if __name__ == '__main__':
    main()
