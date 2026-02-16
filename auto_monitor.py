import argparse
import time

from database import init_db
from monitoring import monitor_all_customers


def run(interval_seconds: int, limit: int | None) -> None:
    init_db()
    print(f"Auto monitor started. interval={interval_seconds}s limit={limit}")
    while True:
        summary = monitor_all_customers(limit=limit)
        print(
            f"[monitor] checked={summary['customers_checked']} "
            f"alerts={summary['alerts_generated']}"
        )
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Continuous monitoring scheduler.")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="0 means all customers")
    args = parser.parse_args()
    run(args.interval_seconds, None if args.limit <= 0 else args.limit)


if __name__ == "__main__":
    main()
