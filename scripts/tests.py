import json
import os
from gnn_scheduler.utils import get_data_path

print(get_data_path())

downloaded_file_path = (
    get_data_path() / "raw" / "small_random_instances_0.json"
)
downloaded_file_path_str = str(get_data_path())
downloaded_file_path_str = os.path.join(
    downloaded_file_path_str, "raw", "small_random_instances_0.json"
)

assert str(downloaded_file_path) == downloaded_file_path_str


def fix_json_file(file_path):
    with open(
        file_path, "r", encoding="utf-8-sig"
    ) as f:  # utf-8-sig handles BOM
        content = f.read()

    # Find where the actual JSON content starts (typically after HTML content)
    json_start = content.find("[") if "[" in content else content.find("{")

    if json_start != -1:
        print(f"{json_start =}")
        json_content = content[json_start:]
        # Find where JSON ends
        last_bracket = (
            json_content.rfind("]")
            if "[" in content
            else json_content.rfind("}")
        )
        print(f"{last_bracket =} / {len(json_content) =}")
        if last_bracket != -1:
            json_content = json_content[: last_bracket + 1]

        # Save the fixed content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_content)
        print(f"Fixed JSON file: {file_path}")
    else:
        print(f"Could not find JSON content in file: {file_path}")


def check_for_bom(file_path):
    with open(file_path, "rb") as f:  # Open in binary mode
        bom = f.read(4)  # Read first 4 bytes

    # UTF-8 BOM: EF BB BF
    if bom.startswith(b"\xef\xbb\xbf"):
        return "UTF-8 BOM detected"

    # UTF-16 BE BOM: FE FF
    elif bom.startswith(b"\xfe\xff"):
        return "UTF-16 BE BOM detected"

    # UTF-16 LE BOM: FF FE
    elif bom.startswith(b"\xff\xfe"):
        return "UTF-16 LE BOM detected"

    # UTF-32 BE BOM: 00 00 FE FF
    elif bom.startswith(b"\x00\x00\xfe\xff"):
        return "UTF-32 BE BOM detected"

    # UTF-32 LE BOM: FF FE 00 00
    elif bom.startswith(b"\xff\xfe\x00\x00"):
        return "UTF-32 LE BOM detected"

    return "No BOM detected"


# Usage
print(check_for_bom(downloaded_file_path_str))

fix_json_file(downloaded_file_path_str)


with open(downloaded_file_path_str, "r", encoding="utf-8") as f:
    content = f.read()
    print("First 100 characters:", content[:100])
    # Then try parsing
    try:
        json_data = json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON Error:", e)
        print(
            "Character at error position:",
            repr(content[e.pos - 5 : e.pos + 5]),
        )
