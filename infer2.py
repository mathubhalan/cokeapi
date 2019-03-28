import numpy as np
import tensorflow as tf
import cv2 as cv
import json
import glob


# Read the graph.
with tf.gfile.FastGFile('/home/admin/adcontext/finetuned_model/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print("in 1")

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    print("in 2")

    filepaths = [f for f in glob.glob("/home/admin/adcontext/training_demo/images/CokeAccuracyEvaluation/**/*.jpg")]

    # Read and preprocess an image.
    for image_path in filepaths:
        print (image_path)

       	#filename = 'C:/MK/03_Others/CNN/adcontext/Images/CokeAccuracyEvaluation/1RightImage/00000017-PHOTO-2018-05-31-10-44-35.jpg'
		#filename = 'C:/MK/03_Others/CNN/adcontext/Images/CokeAccuracyEvaluation/5BadQualityImage/00000075-PHOTO-2018-05-31-16-54-26.jpg'
        img = cv.imread(image_path)
        out_image_path = image_path.replace("CokeAccuracyEvaluation","CokeAccuracyResults")

        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

		# Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),sess.graph.get_tensor_by_name('detection_scores:0'),sess.graph.get_tensor_by_name('detection_boxes:0'),sess.graph.get_tensor_by_name('detection_classes:0')],feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

		# Visualize detected bounding boxes.
        num_detections = int(out[0][0])

        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                height = int(y) - int(bottom)
                width = int(right) - int(x)
                ratio = width / height
                print ('Input - {}, Number of detection - {}, Detection - {}, Class ID - {}, Score - {}, Height - {}, Width - {}, Ratio - {}, Output - {}'.format (image_path, str(num_detections), str( i),str(classId),str(score),height, width, ratio,out_image_path))
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                text = "Coke" + str(i)
                cv.putText(img, text, (int(x),  int(y)), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv.LINE_AA) 
        cv.imwrite(out_image_path,img)
