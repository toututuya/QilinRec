from src.data import prepare
from src.runtime import configure

if __name__ == "__main__":
    args, cfg = configure("Prepare temporally separated Qilin requests and content features")
    if not args.data:
        raise SystemExit("Supply --data or QILIN_DATA")
    print(prepare(args.data, args.run, cfg), flush=True)
