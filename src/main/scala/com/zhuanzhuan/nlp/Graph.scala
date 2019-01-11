/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.zhuanzhuan.nlp

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{Vector => BV}

private[nlp] class Node extends Serializable {
  var x: Int = 0  //输入序列第x个值
  var y: Int = 0  //第x个输入对应的输出y
  var alpha: Double = 0.0  //forward-backward算法中，forward阶段的alpha值
  var beta: Double = 0.0  //backward阶段中保存的beta值
  var cost: Double = 0.0  //节点的cost，定义为当输出标签为y时，w*phi(y,x)，由于phi每个元素只会返回0或1，
                                  //那么计算cost可以简化为，计算fvector中第y列对应的权重之和
  var bestCost: Double = 0.0  //从起始节点到当前节点最好的cost，用于viterbi算法
  var prev: Option[Node] = None  //该节点上一个连接节点，也是用于viterbi算法
  var fVector: Int = 0  //该节点对应的feature向量，对应于featureCache的某一行
  val lPath: ArrayBuffer[Path] = new ArrayBuffer[Path]()  //左边的path集合
  val rPath: ArrayBuffer[Path] = new ArrayBuffer[Path]()  //右边的path集合

  /**
    * simplify the log likelihood.
    */
  def logSumExp(x: Double, y: Double, flg: Boolean): Double = {
    val MINUS_LOG_EPSILON = 50.0
    if (flg) y
    else {
      val vMin: Double = math.min(x, y)
      val vMax: Double = math.max(x, y)
      if (vMax > vMin + MINUS_LOG_EPSILON) vMax else vMax + math.log(math.exp(vMin - vMax) + 1.0)
    }
  }

  def calcAlpha(nodes: ArrayBuffer[Node]): Unit = {
    alpha = 0.0
    var i = 0
    while (i < lPath.length) {
      alpha = logSumExp(alpha, lPath(i).cost + nodes(lPath(i).lNode).alpha, i == 0)
      i += 1
    }
    alpha += cost
  }

  def calcBeta(nodes: ArrayBuffer[Node]): Unit = {
    beta = 0.0
    var i = 0
    while (i < rPath.length) {
      beta = logSumExp(beta, rPath(i).cost + nodes(rPath(i).rNode).beta, i == 0)
      i += 1
    }
    beta += cost
  }

  def calExpectation(
      expected: BV[Double],
      Z: Double,
      size: Int,
      featureCache: ArrayBuffer[Int],
      nodes: ArrayBuffer[Node]): Unit = {
    val c: Double = math.exp(alpha + beta - cost - Z)

    var idx: Int = fVector
    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + y) += c
      idx += 1
    }

    var i = 0
    while (i < lPath.length) {
      lPath(i).calExpectation(expected, Z, size, featureCache, nodes)
      i += 1
    }
  }

}

private[nlp] class Path extends Serializable {
  var rNode: Int = 0  //边右边连接的节点
  var lNode: Int = 0  //边左边连接的节点
  var cost: Double = 0.0  //cost定义为：w * phi(y1, y2, x)， 由于phi函数返回的向量每个元素取值0或1，
                                  //那么可以简化为计算fvector中左节点为y1，右节点为y2对应权重之和。
  var fVector: Int = 0  //对应的Bigram提取的Feature向量，对应于featureCache某一行

  def calExpectation(
      expected: BV[Double],
      Z: Double,
      size: Int,
      featureCache: ArrayBuffer[Int],
      nodes: ArrayBuffer[Node]): Unit = {
    val c: Double = math.exp(nodes(lNode).alpha + cost + nodes(rNode).beta - Z)
    var idx: Int = fVector

    while (featureCache(idx) != -1) {
      expected(featureCache(idx) + nodes(lNode).y * size + nodes(rNode).y) += c
      idx += 1
    }
  }

  def add(lnd: Int, rnd: Int, nodes: ArrayBuffer[Node]): Unit = {
    lNode = lnd
    rNode = rnd
    nodes(lNode).rPath.append(this)
    nodes(rNode).lPath.append(this)
  }
}
