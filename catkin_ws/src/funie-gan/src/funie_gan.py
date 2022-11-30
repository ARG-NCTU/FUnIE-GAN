#!/usr/bin/env python3

import cv2
import numpy as np
from os.path import join, exists
import tensorflow as tf
from keras.models import model_from_json
## local libs
from data_utils import preprocess, deprocess

import rospy
from sensor_msgs.msg import CompressedImage


class FUnIE_GAN():
    def __init__(self):
        model_name = rospy.get_param('~model_name')
        
        checkpoint_dir = "/home/arg/FUnIE-GAN/TF-Keras/models/gen_p/"
        model_h5 = checkpoint_dir + model_name + ".h5"  
        model_json = checkpoint_dir + model_name + ".json"

        # sanity
        assert (exists(model_h5) and exists(model_json))

        # load model
        self.graph = tf.get_default_graph()
        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        self.funie_gan_generator = model_from_json(loaded_model_json)
        # load weights into new model
        self.funie_gan_generator.load_weights(model_h5)
        print("\nLoaded model")

        self.last_image = None

        # Subscriber
        self.image_sub = rospy.Subscriber("unity/under_water/compressed", CompressedImage, self.sub_img, queue_size=1)

        # Publisher
        self.pub_result = rospy.Publisher("funie_gan/result/compressed", CompressedImage, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.05), self.cb_timer) # Doing callback function at 20Hz


    def sub_img(self, data):
        self.last_image = data

    def cb_timer(self, event):
        try:
            image = self.last_image
            np_arr = np.fromstring(image.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # # inference
            inp_img = cv2.resize(image_np, (256, 256)).astype(np.float32)
            im = preprocess(inp_img)
            im = np.expand_dims(im, axis=0) # (1,256,256,3)
            with self.graph.as_default():
                gen = self.funie_gan_generator.predict(im)
            gen_img = deprocess(gen)[0].astype('uint8')

            # Publish detection result
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', gen_img)[1]).tostring()
            self.pub_result.publish(msg)

        except Exception as e:
            print(e)
            pass

    
if __name__=="__main__":
    rospy.init_node("funie_gan", anonymous=True)
    funie_gan = FUnIE_GAN()
    rospy.spin()
