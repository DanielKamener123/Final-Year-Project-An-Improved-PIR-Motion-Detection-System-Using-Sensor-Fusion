import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

#POSITIVE_LABELS = {"human", "human present", "person", "person present", "1"}

# The Converted Video That Was Labelled In Label Studio
LABELSTUDIO_VIDEO = "video_labelstudio.mp4"

# Get All Labels Needed From The Label Studio Export
def labelsFromValue(value):
    labels = []
    for key in ("timelinelabels", "videotimelinelabels", "labels", "choices"):
        item = value.get(key)
        if isinstance(item, list):
            labels.extend(item)
        elif item is not None:
            labels.append(item)
    return labels


def extractFrameRanges(exportData, taskId=None):
    ranges = []
    tasks = exportData if isinstance(exportData, list) else [exportData]

    if taskId is not None:
        tasks = [task for task in tasks if str(task.get("id")) == str(taskId)]
        if not tasks:
            raise ValueError(f"Could not find Label Studio task ID {taskId}")
    elif len(tasks) > 1:
        raise ValueError(
            "The Label Studio export contains multiple tasks. "
            "Use --task-id to select the task for this recording."
        )

    for task in tasks:
        annotations = task.get("annotations") or task.get("completions") or []
        for annotation in annotations:
            for result in annotation.get("result", []):
                value = result.get("value", {})
                labels = labelsFromValue(value)
                isHuman = "human" in labels
                if not isHuman:
                    continue

                for item in value.get("ranges", []):
                    start = item.get("start")
                    end = item.get("end")
                    if start is not None and end is not None and float(end) >= float(start):
                        ranges.append((int(round(float(start))), int(round(float(end))), 1))

    return ranges


def chooseFrameBase(frameRanges, requestedBase):
    if requestedBase in (0, 1):
        return requestedBase
    if any(start == 0 or end == 0 for start, end, _ in frameRanges):
        return 0
    return 1


def frameRangesToTimes(frameRanges, frameTimes, frameBase):
    if frameTimes.empty:
        raise ValueError("video_frames.csv contains no frames")

    if "time_s" not in frameTimes.columns:
        raise ValueError("video_frames.csv must contain the time_s column from the new collector")
    times = frameTimes["time_s"].to_numpy(dtype=np.float64)
    if len(times) >= 2:
        medianPeriod = float(np.median(np.diff(times)))
    else:
        medianPeriod = 1.0 / 30.0

    converted = []
    for startFrame, endFrame, human in frameRanges:
        startIndex = startFrame - frameBase
        endIndex = endFrame - frameBase

        startIndex = max(0, min(startIndex, len(times) - 1))
        endIndex = max(startIndex, min(endIndex, len(times) - 1))

        startTime = float(times[startIndex])
        if endIndex + 1 < len(times):
            endTime = float(times[endIndex + 1])
        else:
            endTime = float(times[endIndex] + medianPeriod)

        converted.append((startTime, endTime, human))

    converted.sort(key=lambda item: item[0])
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio TimelineLabels to labels.csv")
    parser.add_argument("--export", required=True, help="Label Studio JSON export")
    parser.add_argument("--recording", required=True, help="Recording folder containing metadata.json")
    parser.add_argument("--task-id", default=None, help="Label Studio task ID when the export contains more than one task")
    parser.add_argument(
        "--frame-base",
        choices=["auto", "0", "1"],
        default="auto",
        help="Frame numbering used by the Label Studio export. Auto uses 0-based if any range contains frame 0, otherwise 1-based.",
    )
    args = parser.parse_args()

    recordingDir = Path(args.recording)
    metadataPath = recordingDir / "metadata.json"
    labelsPath = recordingDir / "labels.csv"

    with open(metadataPath, "r", encoding="utf-8") as metadataFile:
        metadata = json.load(metadataFile)

    with open(args.export, "r", encoding="utf-8") as exportFile:
        exportData = json.load(exportFile)

    frameTimesPath = recordingDir / metadata.get("video_frames_file", "video_frames.csv")
    frameTimes = pd.read_csv(frameTimesPath)
    frameRanges = extractFrameRanges(exportData, taskId=args.task_id)
    requestedBase = None if args.frame_base == "auto" else int(args.frame_base)
    frameBase = chooseFrameBase(frameRanges, requestedBase)

    lastFrame = max((end for _, end, _ in frameRanges), default=frameBase)
    if lastFrame - frameBase > len(frameTimes) - 1:
        raise ValueError(
            f"A label ends at frame {lastFrame} but the recording only has {len(frameTimes)} frames. "
            'Check the Label Studio config uses frameRate="30".'
        )
    ranges = frameRangesToTimes(frameRanges, frameTimes, frameBase)

    with open(labelsPath, "w", newline="", encoding="utf-8") as labelsFile:
        writer = csv.writer(labelsFile)
        writer.writerow(["start_s", "end_s", "human"])
        writer.writerows(ranges)

    print(f"Detected Label Studio frame numbering base: {frameBase}")
    print(f"Saved {len(ranges)} human-present intervals to {labelsPath}")
    print("All video time outside these intervals will be treated as no-human.")


if __name__ == "__main__":
    main()
