import cv2
from ultralytics import YOLO

PERSON = 0  # COCO class id for person
BALL = 32   # COCO class id for sports ball

WEIGHTS = "yolo11m.pt"  # Medium model, small was unable to capture ball


def track_players(frame, model):
    """
    Tracks people in a frame.

    Args:
        frame (np.ndarray): BGR video frame.
        model (YOLO): Model used ONLY for player tracking (tracker state lives on it).

    Returns:
        dict[int, tuple[int, int, int, int]]: track_id -> (x1, y1, x2, y2).
    """
    results = model.track(
        frame, classes=[PERSON], conf=0.25, imgsz=1280,
        persist=True, verbose=False,
    )[0]

    players = {}
    if results.boxes.id is None:
        return players

    boxes = results.boxes.xyxy.int().tolist()
    ids = results.boxes.id.int().tolist()
    for (x1, y1, x2, y2), tid in zip(boxes, ids):
        players[tid] = (x1, y1, x2, y2)
    return players


def assign_ball(frame, model, players, last_holder=None):
    """
    Detects the ball (no tracking) and assigns possession to the closest player.
    If the ball isn't seen, or isn't near anyone (pass/shot in flight),
    possession stays with the last player it was assigned to.

    Args:
        frame (np.ndarray): BGR video frame.
        model (YOLO): Model used for ball detection. Must be a different instance
            from the tracking model, otherwise its tracker callbacks would run here.
        players (dict[int, tuple]): Output of track_players.
        last_holder (int | None): Previous holder's track id.

    Returns:
        tuple[int | None, tuple | None]: (holder_id, ball_box)
    """
    results = model.predict(frame, classes=[BALL], conf=0.05, imgsz=1920, verbose=False)[0]
    if len(results.boxes) == 0:
        return last_holder, None

    # Most confident ball candidate
    best = int(results.boxes.conf.argmax())
    bx1, by1, bx2, by2 = results.boxes.xyxy[best].int().tolist()
    ball_box = (bx1, by1, bx2, by2)
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2

    nearest, nearest_d, nearest_h = None, float("inf"), 0
    for tid, (x1, y1, x2, y2) in players.items():
        # Distance from ball centre to player box (0 if ball is inside the box)
        dx = max(x1 - cx, 0, cx - x2)
        dy = max(y1 - cy, 0, cy - y2)
        d = (dx * dx + dy * dy) ** 0.5
        if d < nearest_d:
            nearest, nearest_d, nearest_h = tid, d, y2 - y1

    # Only hand over possession if the ball is actually close to that player
    # (scaled by player height so it works near and far from the camera).
    if nearest is not None and nearest_d <= 0.25 * nearest_h:
        return nearest, ball_box
    return last_holder, ball_box


def draw_boxes(VIDEO_IN: str, VIDEO_OUT: str) -> None:
    """
    Orchestrates player tracking + ball possession and writes an annotated video.

    Args:
        VIDEO_IN (str):  Input video path.
        VIDEO_OUT (str): Output video path.
    """
    player_model = YOLO(WEIGHTS)
    ball_model = YOLO(WEIGHTS)  # separate instance so tracker state doesn't leak in

    cap = cv2.VideoCapture(VIDEO_IN)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    holder = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        players = track_players(frame, player_model)
        holder, ball_box = assign_ball(frame, ball_model, players, holder)

        for tid, (x1, y1, x2, y2) in players.items():
            has_ball = tid == holder
            color = (0, 0, 255) if has_ball else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if has_ball else 2)
            label = f"#{tid}" + (" BALL" if has_ball else "")
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)

        cv2.putText(frame, f"Possession: #{holder}" if holder is not None else "Possession: -",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    draw_boxes("play1.mp4", "play1_boxes.mp4")
