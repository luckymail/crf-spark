package com.zhuanzhuan.nlp.dictlabel

import com.zhuanzhuan.jiezi.dict.loader.{PosDictLoader, StopDictLoader, SynDictLoader, WeightDictLoader}
import com.zhuanzhuan.jiezi.io.{HdfsToStream, TableToStream}
import com.zhuanzhuan.jiezi.segment.NShortSegment

class WordSeg extends Serializable {
  val termDictLoader = new PosDictLoader(new TableToStream)
  val stopDictLoader = new StopDictLoader(new TableToStream)
  val nshortDictLoader = new WeightDictLoader(new TableToStream)
  val synonymsDictLoader = new SynDictLoader(new TableToStream)

  val segmenter = new NShortSegment(
    nshortDictLoader.loadDict("hdp_zhuanzhuan_dm_algo.dm_algo_jiezi_nshort_dict/name=zz_default_nshort/version=v1"),
    Array(
      termDictLoader.loadDict("hdp_zhuanzhuan_dm_algo.dm_algo_jiezi_pos_dict/name=ansj_pos/version=v4"),
      termDictLoader.loadDict("hdp_zhuanzhuan_dm_algo.dm_algo_jiezi_pos_dict/name=atomic_pos/version=v4")
      ),
    stopDict = stopDictLoader.loadDict("hdp_zhuanzhuan_dm_algo.dm_algo_jiezi_stop_dict/name=stop/version=v4"),
    Array(
      synonymsDictLoader.loadDict("hdp_zhuanzhuan_dm_algo.dm_algo_jiezi_synonyms_dict/name=zz_default_syn/version=v2")
    )
  )

  def wordSeg(content: String): Array[String] = {
    val wordSegs: Array[String] = segmenter.parse(content).terms.map(x => x.word)
    wordSegs
  }
}