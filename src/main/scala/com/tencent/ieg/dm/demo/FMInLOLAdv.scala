package com.tencent.ieg.dm.demo

import breeze.linalg.DenseVector
import com.tencent.ieg.dm.utils.TDWClient
import org.apache.spark.mllib.util.tencent.OptionParser
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by bourneli on 2017/6/20.
  */
class FMInLOLAdv {

    def splitDBAndTable(info: String): (String, String) = {
        val Array(db, table) = info.split("\\s*::\\s*")
        (db, table)
    }

    def main(args:Array[String]) ={
        // 初始化spark
        val conf = new SparkConf().setAppName("Compute distance")
        val sc = new SparkContext(conf)
        val sparkSession = SparkSession.builder().getOrCreate()
        import sparkSession.implicits._

        // 获取参数
        val options = new OptionParser(args)

        val statisDate = options.getString("statis_date")
        val (srcDB, srcTable) = splitDBAndTable(options.getString("src"))
        val (rstDB, rstTable) = splitDBAndTable(options.getString("dst"))
        val parts = options.getInt("parts",100)
        val gameMetric = new DenseVector(options.getString("game_metric").split(",").map(_.toDouble))
        val tdwUser = options.getString("tdw_user")
        val tdwPassword = options.getString("tdw_password")

        // 初始化tdw客户端
        val client = new TDWClient(sc, sparkSession, tdwUser, tdwPassword)
    }
}
