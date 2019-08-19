# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.append('../')

import tensorflow as tf
from datetime import datetime
from libs.predictor import LSTM, CNN_x, null_model
from libs.helpingFun import *


def runDMM(dataset, modelparam):
    # Initialization
    FeatureSelection = 'dfs'
    Model_2_Select = 'CNN_x'
    rho = 1
    Loss = 1e8
    top = 100

    X, Y = loadData()

    problem = 'classification'

    # mark the location of causals for ROC
    # feature_causal = featCausalVec(X.shape[1], idx)
    # tmp_weights = np.zeros(feature_causal.shape)
    # tmp_rank = np.zeros(feature_causal.shape)

    # Model Parameters
    # [epochs, epochsMLP, learning_rate_L, learning_rate_M, learning_rate_S, lambda_1,
    #  batch_size, n_hidden_1, n_hidden_2, n_hidden, smooth_penalty, keep_p, keep_p2] = modelparam

    epochs = modelparam['epochsModel1']
    epochsMLP = modelparam['epochsModel2']
    learning_rate_L = modelparam['learning_rate_L']
    learning_rate_M = modelparam['learning_rate_M']
    learning_rate_S = modelparam['learning_rate_S']
    lambda_1 = modelparam['lambda_1']
    batch_size = int(modelparam['batch_size'])
    n_hidden = int(modelparam['n_hidden']*X.shape[1])
    n_hidden_1 = n_hidden
    n_hidden_2 = int(n_hidden*modelparam['b'])
    smooth_penalty = modelparam['smooth_penalty']
    keep_p = modelparam['keep_p']
    keep_p2 = modelparam['keep_p2']


    # Display and Save
    display_step = 10

    # Network Parameters
    n_input = X.shape[1]  # data input dimension
    n_output = Y.shape[1]  # data output dimension
    n_samples = X.shape[0]  # number of training samples

    print('Number of features :%s' % X.shape[1])
    print('Number of samples  :%s' % X.shape[0])
    print('Batch size         :%s' % batch_size)
    print('Model1 epochs      :%s' % epochs)
    print('Model2  epochs     :%s\n' % epochsMLP)
    # print('CFW: ' + repr(CFW) + '       datatype:' + repr(datatype))

    # ==============================================================================
    # Define the Model Structure

    tf.reset_default_graph()

    # tf Graph input
    with tf.name_scope('input'):
        x = tf.placeholder("float", [None, n_input], 'x-input')
        y = tf.placeholder("float", [None, n_output], 'y-input')
        y_cnn = tf.placeholder("float", [None, n_output], 'y_cnn')

    # for Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep-prob-1')
        keep_prob2 = tf.placeholder(tf.float32, name='keep-prob-2')

    # for BN-lstm
    training = tf.placeholder(tf.bool)

    # Model 2: For Confounding Correction
    # —————————————————————————————————————————————————————————————————————————————

    if Model_2_Select == 'CNN_x':
        # Current Model: 1-D CNN
        model2top = CNN_x(x, y, problem, n_input, n_output=n_output,
                        keep_prob=keep_prob2, BN=True, training=training, DFS=False)

    if Model_2_Select == 'NULL':
        # Null model:
        model2top = null_model(x, y, problem, n_input, n_output=n_output,
                               keep_prob=keep_prob2, BN=True, training=training, DFS=False)

    with tf.name_scope('Model_2_Training'):
        if Model_2_Select != 'NULL':
            optimizer2top_adam_M = tf.train.AdamOptimizer(learning_rate_M).minimize(model2top.loss)

    # Model 1: For locating causality
    # —————————————————————————————————————————————————————————————————————————————
    model1 = LSTM(x, y, y_cnn, problem, n_input, n_steps=1, n_hidden=n_hidden,
                  n_classes=n_output, keep_prob=keep_prob)

    # var_all = tf.trainable_variables()
    var_no_DFS = [var for var in tf.trainable_variables() if not "DFS_weights" in var.name]
    # var_DFS = [var for var in tf.trainable_variables() if "DFS_weights" in var.name]

    with tf.name_scope('Model_1_Training'):
        optimizer1noDFS_M = tf.train.AdamOptimizer(learning_rate_M).minimize(model1.loss, var_list=var_no_DFS)
        optimizer1_M = tf.train.AdamOptimizer(learning_rate_M).minimize(model1.loss)

    # Main Model: Superposition
    # —————————————————————————————————————————————————————————————————————————————
    if problem == "regression":
        output_pred = model1.pred + model2top.pred
        cost = tf.nn.l2_loss(tf.subtract(output_pred, y))
        loss = cost

    # Training

    t_start = datetime.now()  # timer

    # Create a saver
    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=0.05)

    with tf.Session() as sess:
        global_step_var = 0
        # Check if there exists previously trained model
        ckpt = tf.train.get_checkpoint_state("./save/")
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint("./save/"))
            # Get the global_step value
            global_step_var = int(ckpt.model_checkpoint_path.split('-')[-1])
            print("Start from epoch: ", global_step_var)
            # Get global_step for pretrain part
            if os.path.exists("./save/global_step_pre.csv"):
                global_step_pre = genfromtxt("./save/global_step_pre.csv",
                            delimiter=',').reshape([-1, 1]).astype(int)[-1, 0]
        if global_step_var == 0:
            global_step_pre = 0
            # Initializing all the variables only when `global_step_var == 0`
            # or just restore the previous graph and variables
            tf.global_variables_initializer().run()

        # ==============================================================================
        # Train model2

        if Model_2_Select != 'NULL':

            print("Train Model 2 for %s epochs:" % epochsMLP)
            epoch = 0
            for epoch in range(epochsMLP):
                epoch += 1
                # shuffle the indexs
                idxs = np.arange(0, n_samples)
                np.random.shuffle(idxs)
                # loop through the batchs
                for batch_step in range(n_samples // batch_size):
                    index = idxs[batch_step * batch_size:(batch_step + 1) * batch_size]
                    X_batch = X[index, :]
                    y_batch = Y[index,]

                    # Training MLP
                    sess.run(optimizer2top_adam_M, feed_dict={
                            x: X_batch, y: y_batch, keep_prob2: keep_p2, training: True})


                loss_2top, y_pred_2top = sess.run([model2top.loss, model2top.pred],
                                                      feed_dict={x: X, y: Y, keep_prob2: 1, training: False})

                with open("./save/paramsModel2.csv", "a+") as fp:
                    csv.writer(fp, dialect='excel').writerow((global_step_pre + epoch, loss_2top))
                # prediction of Model2MLP
                with open("./save/y_pred_2top.csv", "a+") as fp:
                    csv.writer(fp, dialect='excel').writerow(y_pred_2top[:, 0])

                if epoch % display_step == 0 or epoch == 1:
                    print("Epoch %6d," % (global_step_pre + epoch), "Loss Model2: %2.5f" % loss_2top)

                if loss_2top < 0.000001:
                    break


            if Model_2_Select == 'MLP':
                h1_mlp = model2top.weights['h1'].eval().reshape([n_input, n_hidden_1])  # all the feature weights
                h1_mlp = np.nan_to_num(h1_mlp)  # to correct NaN
                np.savetxt("./save/Model2MLP_h1.csv", h1_mlp, delimiter=",")
                h2_mlp = model2top.weights['h2'].eval().reshape([n_hidden_1, n_hidden_2])  # all the feature weights
                h2_mlp = np.nan_to_num(h2_mlp)  # to correct NaN
                np.savetxt("./save/Model2MLP_h2.csv", h2_mlp, delimiter=",")

            with open("./save/global_step_pre.csv", "a+") as fp:
                csv.writer(fp, dialect='excel').writerow([global_step_pre + epoch])

            # Yres = sess.run(model2top.yres, feed_dict={
            #         x: X, y: Y, keep_prob2: 1, training: False})


            Y_cnn = sess.run(model2top.pred, feed_dict={
                    x: X, y: Y, keep_prob2: 1, training: False})

        else:
            print("Skip Training on Model-2")
            Y_cnn = Y - Y

        # ==============================================================================
        # Training cycle for LSTM

        print("\nStart Training Cycle:\n")

        for epoch in range(epochs):

            epoch += 1
            idxs = np.arange(0, n_samples)
            np.random.shuffle(idxs)

            for fbacth_step in list(range(1)):
                # loop through the batchs
                for batch_step in range(n_samples // batch_size):
                    index = idxs[batch_step * batch_size:(batch_step + 1) * batch_size]
                    X_batch = X[index, :]
                    y_batch = Y[index, ]
                    y_cnn_batch = Y_cnn[index, ]
                    if FeatureSelection == 'fs':
                        sess.run([optimizer1noDFS_M], feed_dict={
                            x: X_batch, y: y_batch, y_cnn: y_cnn_batch,
                            keep_prob: keep_p, keep_prob2: 1, training: True})
                    elif FeatureSelection == 'dfs':
                        sess.run([optimizer1_M], feed_dict={
                            x: X_batch, y: y_batch, y_cnn: y_cnn_batch,
                            keep_prob: keep_p, keep_prob2: 1, training: True})
                        feature_weights1 = model1.w.eval().reshape([-1, X.shape[1]])[0]  # all the feature weights
                        feature_weights1 = np.nan_to_num(feature_weights1)  # to correct NaN
                        feature_selected1 = (abs(feature_weights1) > 0)

            # ==============================================================================
            # Display logs per epoch step

            if epoch % display_step == 0 or epoch == 1:

                if Model_2_Select == 'NULL':
                    model2_loss, model2_pred = sess.run([model2top.loss, model2top.pred],feed_dict={x: X, y: Y})
                else:
                    model2_loss, model2_pred = sess.run([model2top.loss, model2top.pred ],
                                                        feed_dict={x: X, y: Y, keep_prob2: 1, training: False})

                if problem == 'regression':
                    dict_train_1 = {x: X, y: Y, y_cnn: Y_cnn, keep_prob: keep_p, keep_prob2: 1, training: False}
                    model1_pred, y_pred, model1_loss = sess.run([model1.pred, output_pred, model1.loss], dict_train_1)
                    dict_train_2 = {x: X, y: Y, y_cnn: Y_cnn, keep_prob: keep_p, keep_prob2: 1, training: False}
                    Loss, Cost = sess.run([loss, cost], dict_train_2)
                else:
                    dict_train_1 = {x: X, y: Y, y_cnn: Y_cnn, keep_prob: keep_p, keep_prob2: 1, training: False}
                    model1_pred, y_pred, model1_loss = sess.run([model1.pred, model1.bpred, model1.loss], dict_train_1)

                print("Epoch: %6d \n"
                      "     LSTM Loss: %f\n"
                      "      CNN Loss: %f\n"
                      % (epoch, model1_loss, model2_loss))

                # _____________________________________________________________________________
                if FeatureSelection == 'dfs':
                    # Sparse Control before evaluation
                    feature_weights1 = sparseWeight(feature_weights1)

                    # Weights and Rank

                    #with open("./save/weights-DMM.csv", "a+") as fp:
                    #    csv.writer(fp, dialect='excel').writerow(np.concatenate(
                    #        [[epoch + global_step_var], feature_weights1]))
                    np.save("./save/weights-DMM.npy", feature_weights1)

                    # weights rank
                    with open("./save/weightsRank.csv", "a+") as fp:
                        # ranked features by weights: highest->lowest
                        rank, weights_sorted = featureRank(feature_weights1)
                        csv.writer(fp, dialect='excel').writerow(np.concatenate(
                            [[epoch + global_step_var], rank]))
                        csv.writer(fp, dialect='excel').writerow(np.concatenate(
                            [[epoch + global_step_var], np.array(weights_sorted)]))

                # _____________________________________________________________________________
                # prediction of Main Model
                with open("./save/y_pred.csv", "a+") as fp:
                    csv.writer(fp, dialect='excel').writerow(y_pred[:, 0])
                # prediction of LSTM
                with open("./save/y_lstm_pred.csv", "a+") as fp:
                    csv.writer(fp, dialect='excel').writerow(model1_pred[:, 0])
                # prediction of Model2top with after LSTM
                with open("./save/y_cnn_pred.csv", "a+") as fp:
                    csv.writer(fp, dialect='excel').writerow(model2_pred[:, 0])

                t_delta = datetime.now() - t_start  # timer
                print("\nCurrent Training Time:", t_delta.total_seconds(), "seconds.")

                with open("./save/modelparams.csv", "a+") as fp:
                    if (global_step_var + epoch) == 1:
                        csv.writer(fp, dialect='excel').writerow(["Epoch",
                                                                  "Run Time",
                                                                  "CNN Loss",
                                                                  "LSTM Loss", 
                                                                  "AUC"])
                    csv.writer(fp, dialect='excel').writerow([epoch + global_step_var,
                                                              t_delta.total_seconds(),
                                                              model2_loss,
                                                              model1_loss])

            # Save the model
            if epoch == epochs:
                save_path = saver.save(sess, "./save/model.ckpt", global_step=global_step_var + epoch)
                print('Model saved in the file %s' %save_path)



    # Timing
    t_delta = datetime.now() - t_start  # timer
    runSeconds = t_delta.total_seconds()
    print("\nTraining Time:", runSeconds, "seconds.")

    with open('./save/summary.txt', 'a+') as file:
        file.write("Total Training Time   : " + str(t_delta.total_seconds()) + "seconds.\n")
        file.write("Number of features    : %d\n"
                   "Number of samples     : %d\n"
                   "Epochs                : %d\n"
                   % (n_input, n_samples, epochs + global_step_var))


if __name__ == '__main__':

    modelparams = {
        'epochsModel1': 20,
        'epochsModel2': 1500,
        'learning_rate_L': 0.01,
        'learning_rate_M': 0.001,
        'learning_rate_S': 0.001,
        'lambda_1': 0.0001,
        'batch_size': 128,
        'n_hidden': 0.15,
        'b': 0.4,
        'smooth_penalty': 0.3,
        'keep_p': 1,
        'keep_p2': 0.1
    }

    runDMM(True, modelparams)