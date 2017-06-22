package com.tencent.ieg.dm.demo

import org.apache.spark.mllib.util.tencent.OptionParser
import com.tencent.ieg.dm.utils.TDWClient
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.mllib.linalg.{ Vectors => MLLibVectors}
import org.apache.spark.ml.feature.MinMaxScaler

import org.apache.spark.mllib.fm.FMWithSGD

/**
  * Created by bourneli on 2017/6/20.
  */
object FMInLOLAdv {

    def splitDBAndTable(info: String): (String, String) = {
        val Array(db, table) = info.split("\\s*::\\s*")
        (db, table)
    }

    def main(args:Array[String]) ={
        // 初始化spark
        val conf = new SparkConf().setAppName("LOL User Data")
        val sc = new SparkContext(conf)
        val sparkSession = SparkSession.builder().getOrCreate()
        import sparkSession.implicits._

        // 获取参数
        val options = new OptionParser(args)

        val statisDate = options.getString("statis_date")
        val (srcDB, srcTable) = splitDBAndTable(options.getString("src"))
        val sampleRate = options.getDouble("sample")
        val parts = options.getInt("parts", 100)
        val tdwUser = options.getString("tdw_user")
        val tdwPassword = options.getString("tdw_password")


        // 算法参数
        val randomSeed = options.getInt("seed",12345)
        val miniBatchRate = options.getDouble("batch_rate", 0.001)
        val iterNum = options.getInt("fm_iter", 10)
        val stepSize = options.getDouble("step_size", 0.001)
        val fmK = options.getInt("fm_k", 8)
        val regParam = options.getDouble("fm_reg", 0.001)
        val initStd = options.getDouble("fm_init_std", 0.01)
        val negativeLabelRate = options.getDouble("negative_sub_rate", 0.1)

        // 加载数据
        val client = new TDWClient(sc, sparkSession, tdwUser, tdwPassword)

        val featureSize =  client.tdwSql(srcDB).table(srcTable, Array("p_" + statisDate)).first().size - 4
        println("Feature Size:" + featureSize)

        // 获取原始数据
        val originData = client.tdwSql(srcDB).table(srcTable, Array("p_" + statisDate))
            .sample(false, sampleRate)
            .repartition(parts)
            .map(row => {
                val size = row.length
                val features = Vectors.dense({for(i <- 3 until size-1)
                    yield (row.get(i).toString.toDouble)}.toArray)
                val label  = row.getAs[Long](size-1).toDouble
                (label, features)
            }).toDF("label","features")
            .persist(StorageLevel.MEMORY_AND_DISK)
        originData.show(10,false)

        // 正规化特征
        val norm = new MinMaxScaler()
            .setInputCol("features")
            .setOutputCol("norm_features")
            .fit(originData)
        val normData = norm.transform(originData).persist(StorageLevel.MEMORY_ONLY)
        normData.show(10,false)

        // 将数据转成稀疏向量，并且切分训练和预测数据
        val Array(trainData, testData)  =  normData.select("label","norm_features").map(row => {
            val label = row.getAs[Double]("label")
            val normFeatures = row.getAs[DenseVector]("norm_features")
                .toArray
                .zipWithIndex
                .filter({case(value,_) => value != 0d})
                .map(_.swap)
            val sparseFeatures = MLLibVectors.sparse(featureSize, normFeatures)
            LabeledPoint(label, sparseFeatures)
        }).persist(StorageLevel.MEMORY_AND_DISK)
            .randomSplit(Array(0.7,0.3),seed = randomSeed)

        // 调整正负样本比例
        val subTrainData = trainData.filter(_.label == 1d).union(
            trainData.filter(_.label == 0d).sample(false, negativeLabelRate))
            .persist(StorageLevel.MEMORY_AND_DISK)
            .repartition(parts)
        val trainLabelStat = subTrainData.map(x => (x.label,1)).rdd.reduceByKey(_+_).collect.toMap
        println("Train Label Stat")
        trainLabelStat.foreach(println)

        println("Train Data===============")
        subTrainData.show(10,false)
        println("Test Data================")
        testData.show(10,false)
        val testLabelStat = testData.map(x => (x.label,1)).rdd.reduceByKey(_+_)
        testLabelStat.foreach(println)

        // 训练
        val (fm1,loss) = FMWithSGD.trainWithLoss(subTrainData.rdd, task = 1, numIterations = iterNum,
            stepSize = stepSize, miniBatchFraction = miniBatchRate,dim = (true,true,fmK),
            regParam = (0d,0d, regParam),initStd = initStd)

        println("Loss per each round")
        println("loss")
        println(loss.mkString("\n"))


        // 计算AUC
        val pValueAndLabel = testData.map(testPoint =>
            (fm1.predict(testPoint.features), testPoint.label)
        ).cache
        pValueAndLabel.show(20,false)
        val auc = new BinaryClassificationMetrics (pValueAndLabel.rdd).areaUnderROC

        println(
            s"""
               |AUC: $auc
            """.stripMargin)

        sc.stop()
    }
}
