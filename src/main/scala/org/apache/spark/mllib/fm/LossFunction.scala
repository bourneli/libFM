package org.apache.spark.mllib.fm

/**
  * Loss function in Optimization
  * Created by bourneli on 2017/6/22.
  */
trait LossFunction {

    /**
      * Compute the loss value
      * @param pred predicted value
      * @param label actual value
      * @return loss value
      */
    def compute(pred:Double, label:Double):Double
}

/**
  * Squared error loss function, applied in regression.
  */
class SquaredErrorLoss extends LossFunction {

    def compute(pred:Double, label:Double) = {
        val diff = pred - label
        diff * diff
    }
}

/**
  * Logit Loss, funciton, applied in binary classification.
  */
class LogitLoss extends LossFunction {

    /**
      *
      * @param pred predicted value, pred could be any real number
      * @param label actual value, labels should be -1 or 1
      * @return loss value
      */
    def compute(pred:Double, label:Double) = Math.log(1+Math.exp(-label*pred))

}