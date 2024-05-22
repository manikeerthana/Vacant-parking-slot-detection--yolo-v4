import numpy as np
import cv2 # openCV
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=[12,8]

# check the opencv version
print(cv2.__version__)

import requests
url = 'https://raw.githubusercontent.com/Masterx-AI/Project_Object_Detection_Yolo_V4/main/people_bicycles.jpg'
r = requests.get(url, allow_redirects=True)
open('people_bicycles.jpg', 'wb').write(r.content)
test_img = cv2.imread('people_bicycles.jpg')
img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Plotting the image
def plot_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    
plot_image(img)
plt.show()

test_img = cv2.imread('people_bicycles.jpg')
img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Plotting the image
def plot_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    
plot_image(img)
plt.show()# Converting to blob object

scalefactor = 1.0/255.0
new_size = (416, 416)
blob = cv2.dnn.blobFromImage(test_img, scalefactor, new_size, swapRB=True, crop=False)

class_labels_path = "model/coco.names"
class_labels = open(class_labels_path).read().strip().split("\n")
class_labels

# declare repeating bounding box colors for each class 
# 1st: create a list colors as an RGB string array
# Example: Red, Green, Blue, Yellow, Magenda
class_colors = ["255,0,0","0,255,0","0,0,255","255,155,0","255,0, 255"]
 
#2nd: split the array on comma-separated strings and for change each string type to integer
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
 
#3rd: convert the array or arrays to a numpy array
class_colors = np.array(class_colors)
 
#4th: tile this to get 80 class colors, i.e. as many as the classes(16 rows of 5cols each). 
# If you want unique colors for each class you may randomize the color generation or set them manually
class_colors = np.tile(class_colors,(16,1))

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)
 
for i in range(16):
    line = ""
    for j in range(5):
        class_id = i*5 + j
        class_id_str = str(class_id)
        text = "class" + class_id_str
        colored_text = colored(class_colors[class_id][0], class_colors[class_id][1], class_colors[class_id][2], text)
        line += colored_text
    print(line)

# or select the colors randomly
class_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype="uint8")

# Load the pre-trained model 
yolo_model = cv2.dnn.readNetFromDarknet('model/yolov4.cfg','model/yolov4.weights')

# Read the network layers/components. The YOLO V4 neural network has 379 components. They consist of convolutional layers (conv), rectifier linear units (relu) etc.:
model_layers = yolo_model.getLayerNames()
print("number of network components: " + str(len(model_layers))) 
# print(model_layers)

# extract the output layers in the code that follows:
# - model_layer[0]: returns the index of each output layer in the range of 1 to 379
# - model_layer[0] - 1: corrects  this to the range of 0 to 378
# - model_layers[model_layer[0] - 1]: returns the indexed layer name 
output_layers = [model_layers[model_layer - 1] for model_layer in yolo_model.getUnconnectedOutLayers()]
 
# YOLOv4 deploys the same YOLO head as YOLOv3 for detection with the anchor based detection steps, and three levels of detection granularity. 
print(output_layers)

yolo_model.setInput(blob)
 
# compute the forward pass for the input, storing the results per output layer in a list
obj_detections_in_layers = yolo_model.forward(output_layers)
 
# verify the number of sets of detections
print("number of sets of detections: " + str(len(obj_detections_in_layers)))

class_ids_list = []
boxes_list = []
confidences_list = []

def object_detection_attributes(test_image, obj_detections_in_layers, confidence_threshold):
  # get the image dimensions  
  img_height = test_img.shape[0]
  img_width = test_img.shape[1]
  
  # loop over each output layer 
  for object_detections_in_single_layer in obj_detections_in_layers:
    # loop over the detections in each layer
    for object_detection in object_detections_in_single_layer:  
      # get the confidence scores of all objects detected with the bounding box
      prediction_scores = object_detection[5:]
      # consider the highest score being associated with the winning class
      # get the class ID from the index of the highest score 
      predicted_class_id = np.argmax(prediction_scores)
      # get the prediction confidence
      prediction_confidence = prediction_scores[predicted_class_id]
      
      # consider object detections with confidence score higher than threshold
      if prediction_confidence > confidence_threshold:
        # get the predicted label
        predicted_class_label = class_labels[predicted_class_id]
        # compute the bounding box coordinates scaled for the input image
        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
        (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
        start_x_pt = max(0, int(box_center_x_pt - (box_width / 2)))
        start_y_pt = max(0, int(box_center_y_pt - (box_height / 2)))
        
        # update the 3 lists for nms processing
        # - confidence is needed as a float 
        # - the bbox info has the openCV Rect format
        class_ids_list.append(predicted_class_id)
        confidences_list.append(float(prediction_confidence))
        boxes_list.append([int(start_x_pt), int(start_y_pt), int(box_width), int(box_height)])


score_threshold = 0.5
object_detection_attributes(test_img, obj_detections_in_layers, score_threshold)


score_threshold = 0.5
nms_threshold = 0.4
winner_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, score_threshold, nms_threshold)

# loop through the final set of detections remaining after NMS and draw bounding box and write text
for winner_id in winner_ids:
    max_class_id = winner_id
    box = boxes_list[max_class_id]
    start_x_pt = box[0]
    start_y_pt = box[1]
    box_width = box[2]
    box_height = box[3]
    
    #get the predicted class id and label
    predicted_class_id = class_ids_list[max_class_id]
    predicted_class_label = class_labels[predicted_class_id]
    prediction_confidence = confidences_list[max_class_id]
 
    #obtain the bounding box end coordinates
    end_x_pt = start_x_pt + box_width
    end_y_pt = start_y_pt + box_height
    
    #get a random mask color from the numpy array of colors
    box_color = class_colors[predicted_class_id]
    
    #convert the color numpy array as a list and apply to text and box
    box_color = [int(c) for c in box_color]
    
    # print the prediction in console
    predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
    print("predicted object {}".format(predicted_class_label))
    
    # draw rectangle and text in the image
    cv2.rectangle(test_img, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 2)
    cv2.putText(test_img, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
test_imgz = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

plt.imshow(test_imgz)
plt.show()
