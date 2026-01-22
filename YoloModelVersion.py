# https://onlinelibrary.wiley.com/doi/10.1111/ijlh.14455
# Dr. Vanderboom gave me a link to Jansen's paper doing this for VwF gels

pip install ultralytics opencv-python



from ultralytics import YOLO
import cv2

# ------------------------------------------------------------
# Load YOLOv8 segmentation model
# ------------------------------------------------------------
# Options:
# yolov8n-seg.pt  (fast, lower accuracy)
# yolov8s-seg.pt
# yolov8m-seg.pt
# yolov8l-seg.pt
model = YOLO("yolov8n-seg.pt")

# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------
img_path = "image.jpg"  # or video path
results = model(img_path, conf=0.4)

# ------------------------------------------------------------
# ------------------------------------------------------------
for r in results:
    img = r.orig_img

    # Bounding boxes
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    # Segmentation masks (N, H, W)
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
    else:
        masks = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cls = classes[i]
        score = scores[i]
        label = model.names[cls]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.putText(img, f"{label} {score:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Draw segmentation mask
        if len(masks) > 0:
            mask = masks[i]
            mask = (mask * 255).astype("uint8")
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)

    cv2.imshow("YOLOv8 Detection + Segmentation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
