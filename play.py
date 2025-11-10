import cv2
import numpy as np

# --- CONFIG ---
screenshot_path = "screenshot.png"  # change to your test screenshot
judging_line_y = 700  # adjust according to your screenshot/game window
min_note_area = 30  # reduced for better detection of smaller notes
min_aspect_ratio = 0.3  # minimum width/height ratio for notes
max_aspect_ratio = 3.0  # maximum width/height ratio for notes
hold_note_min_height = 50  # minimum height to be considered a hold note
contrast_threshold = 30  # minimum color difference from background

# --- LOAD IMAGE ---
frame = cv2.imread(screenshot_path)
if frame is None:
    raise FileNotFoundError(f"Cannot find {screenshot_path}")

# Convert to grayscale for contrast detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Calculate average background color (assuming most of the image is background)
avg_bg_color = np.mean(gray)

# Create a mask for areas with sufficient contrast from background
_, contrast_mask = cv2.threshold(cv2.absdiff(gray, avg_bg_color), 
                               contrast_threshold, 255, cv2.THRESH_BINARY)
contrast_mask = contrast_mask.astype(np.uint8)

# Apply morphological operations to clean up the mask
kernel = np.ones((3, 3), np.uint8)
contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_OPEN, kernel, iterations=1)
contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(contrast_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get average note width for reference
temp_widths = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if min_note_area < w * h < 10000:  # Filter out very large areas
        temp_widths.append(w)

average_note_width = np.median(temp_widths) if temp_widths else 30

# Process contours
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect_ratio = w / float(h) if h != 0 else 0
    
    # Skip if too small or wrong aspect ratio
    if (area < min_note_area or 
        not (min_aspect_ratio < aspect_ratio < max_aspect_ratio)):
        continue
    
    # Check if this is a hold note (taller than normal)
    is_hold = h > hold_note_min_height
    
    # Get the note color from the original image
    roi = frame[y:y+h, x:x+w]
    avg_color = np.mean(roi, axis=(0, 1))
    
    # Determine note type based on position and size
    if is_hold:
        note_type = "HOLD"
        color = (0, 255, 255)  # Yellow for hold notes
    else:
        # Check if this is a normal note (similar width to average)
        if 0.7 * average_note_width < w < 1.3 * average_note_width:
            note_type = "NORMAL"
            color = (0, 255, 0)  # Green for normal notes
        else:
            continue  # Skip if not matching expected note dimensions
    
    # Draw rectangle and add text
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, note_type, (x, y-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw center point
    cx, cy = x + w // 2, y + h // 2
    cv2.circle(frame, (cx, cy), 3, color, -1)

# --- SAVE/SHOW RESULT ---
output_path = "output_detected_notes.png"
cv2.imwrite(output_path, frame)
print(f"Output saved to {output_path}")

# Show the results
cv2.imshow("Detected Notes", frame)
cv2.imshow("Contrast Mask", contrast_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
