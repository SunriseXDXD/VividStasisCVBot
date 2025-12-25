import cv2
import numpy as np
from mss import mss
import time
from KeyboardIO import KeyboardIO

# Initialize keyboard handler
keyboard = KeyboardIO()

# Track active hold notes
active_holds = {
    'left': False,  # Track if a hold note is active on left side
    'right': False  # Track if a hold note is active on right side
}

# Track pressed keys to avoid redundant key events
pressed_keys = set()

def classify_note_color(bgr_roi):
    if bgr_roi.size == 0:
        return (255, 255, 255)
    roi_hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(roi_hsv.reshape(-1, 3), axis=0)
    if s < 40 and v > 180:
        return (255, 255, 255)
    if h < 10 or h > 170:
        return (0, 0, 255)
    if 90 <= h <= 140:
        return (255, 0, 0)
    b, g, r = np.mean(bgr_roi.reshape(-1, 3), axis=0)
    if r > b and r > g:
        return (0, 0, 255)
    if b > r and b > g:
        return (255, 0, 0)
    return (255, 255, 255)


# --- CONFIG ---
screenshot_path = "screenshot.png"
judging_line_y = 700  # Y-coordinate of the judgment line
judging_line_tolerance = 20  # Pixel tolerance for note detection
min_note_area = 30  # reduced for better detection of smaller notes
min_aspect_ratio = 2.7  # minimum width/height ratio for notes
max_aspect_ratio = 2.9  # maximum width/height ratio for notes
hold_note_min_height = 50  # minimum height to be considered a hold note
contrast_threshold = 30  # minimum color difference from background

# Screen dimensions (update these to match your game window)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Note track positions (x-coordinates)
TRACK_POSITIONS = {
    'd': SCREEN_WIDTH // 4 - 50,      # Leftmost track
    'f': SCREEN_WIDTH // 4 + 50,      # Left center track
    'j': SCREEN_WIDTH * 3 // 4 - 50,  # Right center track
    'k': SCREEN_WIDTH * 3 // 4 + 50   # Rightmost track
}

def get_closest_track(x_pos):
    """Determine the closest track for a given x position"""
    closest_track = None
    min_distance = float('inf')
    
    for track, pos in TRACK_POSITIONS.items():
        distance = abs(x_pos - pos)
        if distance < min_distance:
            min_distance = distance
            closest_track = track
    
    return closest_track

def process_frame(frame, debug_mode=False):
    """Process a single frame to detect notes and handle key presses

    Args:
        frame: Input frame to process
        debug_mode: If True, returns additional debug information

    Returns:
        tuple: (processed_frame, contrast_mask, [detected_notes] if debug_mode else [])
    """
    # Convert to grayscale for contrast detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate average background color
    avg_bg_color = np.mean(gray)

    # Create a mask for areas with sufficient contrast from background
    _, contrast_mask = cv2.threshold(cv2.absdiff(gray, avg_bg_color),
                                    contrast_threshold, 255, cv2.THRESH_BINARY)
    contrast_mask = contrast_mask.astype(np.uint8)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track which keys need to be pressed in this frame
    keys_to_press = set()

    # Store detected notes for debug output
    detected_notes = []

    # Process each detected contour
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h) if h != 0 else 0

        # Skip if too small or wrong aspect ratio
        if (area < min_note_area or
                not (min_aspect_ratio < aspect_ratio < max_aspect_ratio)):
            continue

        # Check if note is near the judgment line
        note_bottom = y + h
        distance_to_line = abs(note_bottom - judging_line_y)

        # For debugging: print note position and distance to line
        if debug_mode:
            print(f"Note at ({x}, {y}) - Size: {w}x{h} - Bottom: {note_bottom} - Dist to line: {distance_to_line}")

        if distance_to_line > judging_line_tolerance:
            if debug_mode:
                print(f"  -> Too far from judgment line (tolerance: {judging_line_tolerance}px)")
            continue

        # Get the note color
        roi = frame[y:y+h, x:x+w]
        color = classify_note_color(roi)

        # Determine if it's a hold note
        is_hold = h > hold_note_min_height

        # Get the track this note is on
        note_center_x = x + w // 2
        track = get_closest_track(note_center_x)

        # Add to keys to press
        if track:
            keys_to_press.add(track)

        # Store note information
        note_info = {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'center_x': note_center_x,
            'center_y': y + h // 2,
            'track': track,
            'type': 'HOLD' if is_hold else 'NORMAL',
            'color': 'RED' if color == (0, 0, 255) else 'BLUE' if color == (255, 0, 0) else 'UNKNOWN',
            'distance_to_line': distance_to_line
        }
        detected_notes.append(note_info)

        # Draw debug info
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        note_type = note_info['type']
        cv2.putText(frame, f"{note_type} {track.upper()}",
                   (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw center point and coordinates
        center = (note_info['center_x'], note_info['center_y'])
        cv2.circle(frame, center, 3, (0, 255, 255), -1)
        coord_text = f"({center[0]}, {center[1]})"
        cv2.putText(frame, coord_text,
                   (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Update keyboard state
    keyboard.update({k: k in keys_to_press for k in ['d', 'f', 'j', 'k']})

    # Draw judgment line
    cv2.line(frame, (0, judging_line_y), (frame.shape[1], judging_line_y), (0, 255, 0), 2)

    if debug_mode:
        return frame, contrast_mask, detected_notes
    return frame, contrast_mask

def print_note_info(notes):
    """Print detailed information about detected notes"""
    if not notes:
        print("No notes detected in the image.")
        return
        
    print("\n=== DETECTED NOTES ===")
    print(f"{'Index':<6} {'Type':<6} {'Track':<6} {'Position':<20} {'Size':<15} {'Color':<8} {'Dist to Line'}")
    print("-" * 80)
    
    for i, note in enumerate(notes, 1):
        pos_str = f"({note['center_x']}, {note['center_y']})"
        size_str = f"{note['width']}x{note['height']}"
        print(f"{i:<6} {note['type']:<6} {note['track'].upper():<6} {pos_str:<20} {size_str:<15} {note['color']:<8} {note['distance_to_line']:.1f}px")
    
    print(f"\nTotal notes detected: {len(notes)}\n")

def main():
    global judging_line_y
    
    # For screenshot mode
    if screenshot_path:
        frame = cv2.imread(screenshot_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot find {screenshot_path}")
        
        # Get frame dimensions and auto-adjudt judging_line_y if needed
        h, w = frame.shape[:2]
        if judging_line_y >= h:
            print(f"Warning: judging_line_y ({judging_line_y}) is larger than image height ({h}). Adjusting...")
            judging_line_y = h - 50  # Place 50px from bottom
            print(f"New judging_line_y: {judging_line_y}")
        
        # Process frame with debug information
        print("Processing frame with debug information...")
        print(f"Image size: {w}x{h}, Judging line: y={judging_line_y}, Tolerance: ±{judging_line_tolerance}px")
        
        frame, mask, detected_notes = process_frame(frame, debug_mode=True)
        
        # Print detailed note information
        print_note_info(detected_notes)
        
        # If no notes detected, show debug info
        if not detected_notes:
            print("\nDEBUG: No notes detected. Possible issues:")
            print("1. The judging_line_y may be set incorrectly")
            print("2. The aspect ratio or area filters may be too strict")
            print("3. The contrast threshold may need adjustment")
            print(f"Current settings: min_aspect_ratio={min_aspect_ratio}, max_aspect_ratio={max_aspect_ratio}")
            print(f"min_note_area={min_note_area}, contrast_threshold={contrast_threshold}")
        
        # Save and show results
        output_path = "output_detected_notes.png"
        cv2.imwrite(output_path, frame)
        cv2.imwrite("output_mask.png", mask)
        
        print(f"Results saved to {output_path}")
        print("Close the image window to exit...")
        
        cv2.imshow("Detected Notes (Press 'q' to close)", frame)
        cv2.imshow("Contrast Mask", mask)
        
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        # For real-time mode
        with mss() as sct:
            monitor = {"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
            
            print("Starting note detection. Press 'q' to quit.")
            
            while True:
                # Capture screen
                screenshot = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR)
                
                # Process frame
                frame, _ = process_frame(frame)
                
                # Show the frame
                # cv2.imshow("Rhythm Game Bot", frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
    
    # Clean up
    keyboard.release_all()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        time.sleep(3)
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        keyboard.release_all()
