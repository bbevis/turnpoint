"""Convert the public CaSiNo human negotiations into TurnPoint CSVs.

Run: python test/prepare_casino.py data/casino_raw/casino.json data/casino_formatted
Downloads the pinned original JSON if the source path is missing. Source: Chawla
et al. (2021), https://github.com/kushalchawla/CaSiNo (CC BY 4.0).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "2f6ed4a6a55110152a7699fbaa8150d6036314be"
URL = f"https://raw.githubusercontent.com/kushalchawla/CaSiNo/{REVISION}/data/casino.json"
SHA256 = "4f2c4560a0070906ed018c3f0766e35f8f8f31b36274ebf35b608621915744ab"
SATISFACTION = {name: i for i, name in enumerate([
    "Extremely dissatisfied", "Slightly dissatisfied", "Undecided", "Slightly satisfied", "Extremely satisfied"], 1)}
LIKING = {name: i for i, name in enumerate([
    "Extremely dislike", "Slightly dislike", "Undecided", "Slightly like", "Extremely like"], 1)}
EVENTS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}


def prepare_casino(source, destination):
    """Keep human text turns and supplied individual outcomes; sum joint points.

    Excludes interface events, not natural-language acceptance/rejection. Original
    message indices remain turn numbers. Ratings are explicitly ordered 1--5 and
    modelled as continuous. Joint points are the sum of supplied participant scores,
    not a claim about value created above a disagreement baseline. Demographics,
    private preferences, strategy annotations and deal task_data are not predictors.
    """
    source, destination = Path(source), Path(destination)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(urllib.request.urlopen(URL).read())
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Source differs from pinned CaSiNo release")
    data = json.loads(source.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    tables = {"turns": [], "individual_outcomes": [], "conversation_outcomes": []}
    removed = 0
    for conversation in data:
        identity = f"casino:{conversation['dialogue_id']}"
        participants = conversation['participant_info']
        points = []
        for speaker, participant in participants.items():
            outcome = participant['outcomes']
            points.append(float(outcome['points_scored']))
            tables['individual_outcomes'].append({"conversation_id": identity, "speaker_id": speaker,
                "points_scored": outcome['points_scored'], "satisfaction": SATISFACTION[outcome['satisfaction']],
                "opponent_likeness": LIKING[outcome['opponent_likeness']]})
        tables['conversation_outcomes'].append({"conversation_id": identity, "joint_points": sum(points)})
        seen = set()
        for number, message in enumerate(conversation['chat_logs'], 1):
            if message['text'] in EVENTS:
                removed += 1
                continue
            if not message['text'].strip() or message['id'] not in participants:
                raise ValueError(f"Invalid dialogue text/speaker: {identity}, turn {number}")
            seen.add(message['id'])
            tables['turns'].append({"conversation_id": identity, "turn_number": number,
                                   "speaker_id": message['id'], "text": message['text']})
        if seen != set(participants):
            raise ValueError(f"Missing participant dialogue: {identity}")
    for name, rows in tables.items():
        with (destination / f"{name}.csv").open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {"source_url": URL, "source_sha256": SHA256, "license": "CC BY 4.0",
               "conversations": len(data), "turns": len(tables['turns']), "interface_events_removed": removed,
               "satisfaction_mapping": SATISFACTION, "opponent_likeness_mapping": LIKING,
               "joint_points_definition": "sum of supplied individual points_scored"}
    (destination/'preparation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare_casino(args.source, args.destination)
