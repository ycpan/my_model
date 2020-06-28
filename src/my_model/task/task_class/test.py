
        self.embedded_chars = self.input_x
        self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)
 
	    # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):# "filter_sizes", "3,4,5",
            
            with tf.name_scope("conv-maxpool-%s" % filter_size):
	            # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters] # num_filters= 200
                # filter_shape =[height, width, in_channels, output_channels]
 
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expended,
		                            W,
		                            strides=[1,1,1,1],
		                            padding="VALID",
		                            name="conv")
		        # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
		        # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
		        h,
		        ksize=[1, sequence_length - filter_size + 1, 1, 1],
		        strides=[1,1,1,1],
		        padding="VALID",
		        name="pool")
                pooled_outputs.append(pooled)
 
	    # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
 
	    # Add dropout
        with tf.name_scope("dropout"):
	        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
	    
	    # Final (unnomalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
		                            "W",
		                            shape = [num_filters_total, num_classes],
                                    initializer = tf.contrib.layers.xavier_initializer())
            
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name = "b"))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
 
	    # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
————————————————
版权声明：本文为CSDN博主「pan_jinquan」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/guyuealian/article/details/83995519
