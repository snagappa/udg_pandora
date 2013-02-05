# -*- coding: utf-8 -*-

import numpy as np

import tf as _tf_
transformations = _tf_.transformations
from geometry_msgs.msg import TransformStamped as _TransformStamped_

# Transformer to manage transforms
_LocalTransformer_ = _tf_.Transformer()

# Wrapper class to redirect TransformListener to use the local transformer
class TransformListener():
    def __init__(self):
        self._transformer_ = _LocalTransformer_
        self._transformations_ = transformations
        # Message to query the local transformer
        #self._msg_ = _TransformStamped_()
    
    def asMatrix(self, target_frame, header):
        translation, rotation = self._transformer_.lookupTransform(
            target_frame, header.frame_id, header.stamp)
        mat44 = np.dot(
            self._transformations_.translation_matrix(translation),
            self._transformations_.quaternion_matrix(rotation))
        return mat44

# Wrapper class to redirect TransformBroadcaster to use the local transformer
class TransformBroadcaster():
    def __init__(self):
        self._transformer_ = _LocalTransformer_
        # Message for updating the local transformer
        self._msg_ = _TransformStamped_()
    
    def sendTransform(self, translation, rotation, time, child, parent):
        msg = self._msg_
        msg.header.stamp = time
        msg.header.frame_id = parent
        msg.child_frame_id = child
        (msg.transform.translation.x, 
         msg.transform.translation.y,
         msg.transform.translation.z) = translation
        (msg.transform.rotation.x, msg.transform.rotation.y,
         msg.transform.rotation.z, msg.transform.rotation.w) = rotation
        self._transformer_.setTransform(msg)
    