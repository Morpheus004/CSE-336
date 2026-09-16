## Perfetto profiling trace command
```bash
py-spy record -o profiling/profile2.json --format chrometrace -- python -m cs336_basics.bpe
```
This generates a trace with the name of the function in traces. Line numbers are not visible in the trace names as they are generated as args for the trace slice.
```python
#!/usr/bin/env python3
"""
Post-process a py-spy chrometrace JSON so each slice's name includes
its source line number (e.g. "train_bpe:63"), making it visible
directly on the Perfetto timeline without clicking into args.

Usage:
    python add_lines.py input.json output.json
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python add_lines.py <input.json> <output.json>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        data = json.load(f)

    for event in data:
        args = event.get("args", {})
        if "line" in args:
            event["name"] = f"{event['name']}:{args['line']}"

    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Wrote {out_path} with line numbers embedded in slice names.")


if __name__ == "__main__":
    main()
```
This script will append line number to the slices' name.
Then you can use this SQL query to sort by cumulated time and see which part of the function took more time.
```SQL
SELECT name, COUNT(*) as calls, SUM(dur)/1e6 as total_ms FROM slice WHERE name LIKE 'train_bpe%' GROUP BY name ORDER BY total_ms DESC
```
Also simply you can select everything in the trace and generate its flame graph in perfetto itself.
