package SparkTest

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object SparkTellDifference extends App {

  // set up Spark Context
  val sparkSession = SparkSession.builder
    .appName("Simple Application")
    .config("spark.master", "local[*]")
    .getOrCreate()

  val sparkContext = sparkSession.sparkContext
                     sparkContext.setLogLevel("ERROR")

  // step 0: estabelecer conjuntos de dados de origem
  val stringsToAnalyse: List[String] = List("Can you tell the difference between Scala & Spark?",
                                            "You will have to look really closely!")
  val stringsToAnalyseRdd: RDD[String] = sparkContext.parallelize(stringsToAnalyse)

  // step 1: dividir frases em palavras
  val wordsList: List[String]   = stringsToAnalyse    flatMap (_.split(" "))
  val wordsListRdd: RDD[String] = stringsToAnalyseRdd flatMap (_.split(" "))

  // step 2: converter palavras em listas de caracteres, criar pares (chave, valor).
  val lettersList: List[(Char,Int)]   = wordsList    flatMap (_.toList) map ((_,1))
  val lettersListRdd: RDD[(Char,Int)] = wordsListRdd flatMap (_.toList) map ((_,1))

  // step 3: contar cartas
  val lettersCount: List[(Char, Int)] = lettersList groupBy(_._1) mapValues(_.size) toList
  val lettersCountRdd: RDD[(Char, Int)] = lettersListRdd reduceByKey(_ + _)

  // step 4: receba as 5 principais letras de suas frases.
  val lettersCountTop5: List[(Char, Int)] = lettersCount sortBy(- _._2) take(5)
  val lettersCountTop5FromRdd: List[(Char, Int)] = lettersCountRdd sortBy(_._2, ascending = false) take(5) toList

  // o resultado
  println(s"Nativo: ${lettersCountTop5}")
  println(s"Usados: ${lettersCountTop5FromRdd}")

  // feito
  sparkSession.stop()
}