import cv2
import numpy as np
import math
from ultralytics import YOLO
# load card segmentation model
card_model = YOLO('./model/segnLLLL.pt')
flipping_model =YOLO('./model/flipping_model.pt')
id_model = YOLO('./model/id_extract.pt')
n_model =YOLO('./model/main_numbers_v2.pt')
alt_model = YOLO('./model/numbers_spare_v1.pt')
def get_card_vertices(src, model):
    """
    The function get the vertices of the card in the image ordered in clockwise order.
    Parameters:
        src (MatLike): The src image.
    Returns:
        vertices : the vertices of the card ordered in clockwise order,  in case of there is no card the output will be (None).
    """
    ordered_corners = None
    results = model(src)
    # print(results)
    if results[0].masks is None:
        return ordered_corners
    mask = results[0].masks.data[0]
    # Convert to binary for segmentation
    binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
    # Resize the binary mask to match the original image dimensions
    binary_mask = cv2.resize(binary_mask, (src.shape[1], src.shape[0]))
    # add extra area to contour
    binary_mask = cv2.dilate(binary_mask, np.ones((10, 10), np.uint8), iterations=1)
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour is the ID card
    card_contour = max(contours, key=cv2.contourArea)
    # Find corners of the card
    epsilon = 0.05 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)
    # Reorder the points to ensure they are in clockwise order
    corners = np.array(approx).reshape(-1, 2)
    ordered_corners = np.zeros_like(corners)
    # Calculate the centroid of the points
    centroid = np.mean(corners, axis=0)
    # Sort the points based on their angle from the centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    try:
    # Reorder the corners
     for i in range(4):
        ordered_corners[i] = corners[sorted_indices[i]]
     return ordered_corners
    except:
        return ordered_corners
def crop_vertices(src, vertices, out_size):
    """
    This function crop card in horizontal.
    Parameters:
        src (MatLike): The src image.
    Returns:
        card (MatLike) : The cropped card image.
    """
    # Reorder the vertices so that it starts with the longest side
        # Calculate the lengths of the four sides of the quadrilateral
    side_lengths = []
    for i in range(4):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 4]
        side_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        side_lengths.append(side_length)
        # Find the index of the longest side
    longest_side_index = side_lengths.index(max(side_lengths))
    reordered_vertices = np.roll(vertices, -longest_side_index, axis=0)
    # Perspective RATIO MODIFYING
    dst_corners = np.array([[0, 0], [out_size[0] - 1, 0], [out_size[0] - 1, out_size[1] - 1], [0, out_size[1] - 1]], dtype='float32')
    try:
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(reordered_vertices.astype('float32'), dst_corners)
        # Apply the perspective transformation
        result = cv2.warpPerspective(src, M, out_size)
        return result
    except:
        return src
def crop_card(src):
    """
    This function takes image for front or back side of national id card then return with cropped national id card with fixed size.
    Parameters:
        src (MatLike): The src image.
    Returns:
        card (MatLike) : The cropped card image, in case of there is no card the output will be (None).
    """
    # get the coordinates of the quadrilateral card card_vertices
    card_vertices = get_card_vertices(src, model= card_model)
    if card_vertices is None:
        return src
    # crop card and put it in the standard size
    card_img = crop_vertices(src, card_vertices, out_size= (840, 530))
    return card_img
def read_full_card(img, model2 = alt_model ) :
    """
    This function take back card image and get national number on the card.
    Parameters:
        src (MatLike)
    Returns:
        data (str)
    """
    imge_back =(img)
    if imge_back is None:
        return {"status": "Please, Re-Take the ID Card"}
    # Crop the image
    # Run inference on the cropped image
    # Run inference on the cropped image
    results = model2(imge_back)
    # Extract the bounding boxes, their corresponding classes, and confidence scores
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf  # Confidence scores associated with each detection

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name, confidence)
    x_coords_with_classes = [
        (box[0], model2.names[int(c)], conf.item()) for box, c, conf in zip(boxes, classes, confidences) if conf > 0.4
    ]
    # Sort this list based on the x-coordinate
    sorted_classes = [
        cls for _, cls, _ in sorted(x_coords_with_classes, key=lambda x: x[0])
    ]
    # Concatenate the sorted class names into a single string
    id_num_str = "".join(sorted_classes)
    result = 'Please, Re-Take the ID Card' if len(id_num_str) != 14 else id_num_str
    print(id_num_str)
     
    return result
def crop_id (img):
    # Load the image you want to run detection on
    image = img.copy()
    # image = crop_card(image)
    results = id_model(image)
    # The results object is a list with a Results object.
    result = results[0]
    if len(result.boxes.xyxy) == 0:
        return img
    # Get the bounding box tensor
    bbox_tensor = result.boxes.xyxy[0]
    # Check if bbox_tensor is indeed a 1D tensor with four values
    if bbox_tensor.ndim == 1 and bbox_tensor.shape[0] == 4:
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox_tensor.cpu().numpy()
        # Crop the image within the bounding box
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        return cropped_image
    else:
        print("Unexpected bbox_tensor dimensions:", bbox_tensor.shape)
        return img
def read_image(img, model = n_model):

    """
    This function take back card image and get national number on the card.
    Parameters:
        src (MatLike)
    Returns:
        data (str)
    """
    imge_back = img.copy()   
    if imge_back is None:
        return {"status": "Please, Re-Take the ID Card"}

    # Run inference on the cropped image
    results = model(imge_back)
    # Extract the bounding boxes, their corresponding classes, and confidence scores
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf  # Confidence scores associated with each detection

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name, confidence)
    x_coords_with_classes = [
        (box[0], model.names[int(c)], conf.item()) for box, c, conf in zip(boxes, classes, confidences) if conf > 0.3
    ]
    # Sort this list based on the x-coordinate
    sorted_classes = [
        cls for _, cls, _ in sorted(x_coords_with_classes, key=lambda x: x[0])
    ]
    # Concatenate the sorted class names into a single string
    id_num_str = "".join(sorted_classes)
    result = 'Please, Re-Take the ID Card' if len(id_num_str) != 14 else id_num_str
    print(id_num_str)
     
    return result

def card_flipper(img):
    # Perform inference
    results = flipping_model(img)
    # Inspect the first element of the results list
    first_result = results[0]
    boxes = results[0].boxes.xywh# Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    x_coords_with_classes = [ (box[0], flipping_model.names[int(c)]) for box, c in zip(boxes, classes) ]
    # Sort this list based on the x-coordinate
    sorted_classes = [ cls for _, cls in sorted(x_coords_with_classes, key=lambda x: x[0]) ]
    if sorted_classes == []:
        return img
    if sorted_classes[0] =='n-b' or sorted_classes[0] == 'n-f' :
        return img
    else:
        img = cv2.rotate(img, cv2.ROTATE_180)
        return img
def read_id(img):
    # Extract segmented ID Card
    card_cropped = crop_card(img)
    # Flip the card
    card_flipped = card_flipper(card_cropped)
    # Try to read the full card first
    # If successful, return the result
    id_cropped = crop_id(card_flipped)
    if id_cropped.any() == 'No ID Placement found':
        return 'Try Again'
    id_cropped = cv2.resize(id_cropped, (416,416))
    ID_NUMBER = read_image(id_cropped)
    # Check if the full card reading is successful
    if ID_NUMBER == "Please, Re-Take the ID Card":
        full_card_result = read_full_card(card_flipped)
        return full_card_result
    else:
        return ID_NUMBER
    