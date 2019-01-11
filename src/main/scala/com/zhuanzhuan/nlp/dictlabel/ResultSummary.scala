package com.zhuanzhuan.nlp.dictlabel

import scala.io.Source

object ResultSummary {
  def main(args: Array[String]): Unit = {
    val dicts: Array[String] = Source.fromFile("ansj_pos-v4.dict").getLines().toArray.map(_.split("\t")(0))
    val newDicts: Array[String]  = Source.fromFile("title_v3.dict").getLines().toArray

    Source.fromFile("dict_label_res").getLines().toArray.filter(_.split("\t").length >= 8).take(1000).distinct
    .map{x => {
      val sentence: String = x.split("\t").map(f => {val tmp = f.split("\\|"); tmp(2) + "/" + tmp(0)}).mkString(",")
      val info = x.split("\t").map(_.split("\\|")(2)).filter(w => !dicts.contains(w) && newDicts.contains(w)).mkString(",")
      sentence + "\t" + info
    }}.foreach(println)

  }
}
