package com.zhuanzhuan.nlp.dictlabel

import org.apache.spark.sql.SparkSession

import scala.io.Source

object DictLabel extends Serializable {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()

    val posDict: Map[String, String] = Source.fromFile("ansj_pos-v4.dict").getLines()
    .map(x => {val ds = x.split("\t"); (ds(0), ds(1).head.toString)}).toMap

    val newDict: Array[String] = Source.fromFile("title_v3.dict").getLines().toArray.diff(posDict.keys.toArray[String])

    val ac: Automaton = {
      val acAutomaton: Automaton = new Automaton()
      newDict.foreach(acAutomaton.addWord("自定义", _))
      acAutomaton.setFailTransitions()
      acAutomaton
    }

    def getPos(word: String): String = {
      posDict.getOrElse(word, "O")
    }

    def dtSplit(text: String): String = {
      if (ac.search(text).isEmpty) "train" else "test"
    }

    def wordFilter(word: String): Boolean = {
      if (word == "" || word == " " || word == "\t" || word.contains("|") || word.contains("\t"))
        false
      else true
    }

    def includeChr(word: String): Boolean = {
      word.matches(".*[a-zA-Z0-9].*")
    }

    import spark.implicits._
    val wordseg: (String => Array[String])  = new WordSeg().wordSeg
    val doc = spark.read.textFile("/home/hdp_ubu_zhuanzhuan/middata/wenping01/crf/docs").map{ text => (text, dtSplit(text))}
    doc.filter(_._2 == "train").filter(x => !includeChr(x._1)).map { case (line, flag) =>
      val words = wordseg(line)
      if (words.length < 8) "NA"
      else words.filter(wordFilter).map { word => word + "|-|" + getPos(word) }.mkString("\t")
    }.filter(x => x != "NA")
    .repartition(1).write.mode("overwrite").text("/home/hdp_ubu_zhuanzhuan/middata/wenping01/crf/dict_label_train")
    doc.filter(_._2 == "test").filter(x => !includeChr(x._1)).map{ case(line, flag) =>
      val words = wordseg(line)
      if (words.length < 8) "NA"
      else words.filter(wordFilter).map{word => word + "|-|" + getPos(word)}.mkString("\t")
    }.filter(x => x != "NA")
    .repartition(1).write.mode("overwrite").text("/home/hdp_ubu_zhuanzhuan/middata/wenping01/crf/dict_label_test")

  }
}
