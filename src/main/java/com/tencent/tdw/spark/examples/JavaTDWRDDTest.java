package com.tencent.tdw.spark.examples;

import com.tencent.tdw.spark.toolkit.api.java.JavaTDWProvider;
import com.tencent.tdw.spark.toolkit.api.java.JavaTDWFunctions;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.HashMap;

/**
 * Created by sharkdtu
 * 2015/9/20.
 */
public class JavaTDWRDDTest {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext(conf);

        HashMap<String, String> hmap = new HashMap<String, String>();
        for (String arg : args) {
            String[] aa = arg.split("=");
            if (aa.length > 1) {
                hmap.put(aa[0], aa[1]);
            }
        }
//        System.out.println("hmap : " + hmap);

        String tdwuser = hmap.get("user");
        String tdwpasswd = hmap.get("passwd");
        String dbName = hmap.get("db");
        String srcTblName = hmap.get("src_tbl");
        String dstTblName = hmap.get("dst_tbl");
        String tdwurl = hmap.get("tdwurl");
        String[] priParts = hmap.get("pri_parts").split(",");

        JavaTDWProvider tdw = new JavaTDWProvider(sc, tdwuser, tdwpasswd, dbName, "tl");

        JavaRDD<String[]> rdd1 = tdw.table(srcTblName, priParts);
        long cnt = rdd1.count();
        System.out.println("cnt : " + cnt);

        // using tdwurl(e.g. "tdw://dbName/tblName/[priParts]/[subParts]")
        JavaRDD<String[]> rdd2 = JavaTDWFunctions.loadTable(sc, tdwurl);
        JavaTDWFunctions.saveToTable(rdd2, tdwurl);
    }
}