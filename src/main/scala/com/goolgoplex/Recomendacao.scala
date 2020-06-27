package com.goolgoplex

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{asc, col, count, desc, explode, split}
import org.apache.spark.sql.functions._
/**
 * fonte de dados : https://grouplens.org/datasets/movielens/
 * @author web2ajax@gmail.com
 *
 */
object Recomendacao extends Serializable {

  def main(args: Array[String]): Unit = {

  val ss = SparkSession.builder()
      .appName("Row Demo")
      .master("local[3]")
      .getOrCreate()

  val sparkContext = ss.sparkContext
  sparkContext.setLogLevel("ERROR")

   //Carregar os dados relativos aos ratings: userId,
   //movieId, rating
   val ratings_DF = (ss.read
     .option("header", "true")
     .option("inferSchema", "true")
     .csv("src\\main\\resources\\ratings.csv")
     .select("userId", "movieId", "rating")
     ).cache()

    //Carregar os dados relativos aos filmes
    //(titles): movieId, title, genres
    val movies_DF = (ss.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src\\main\\resources\\movies.csv")
      ).cache()

    //Carregar os dados relativos às tags: userId, movieId e tag
    val tags_DF = (ss.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src\\main\\resources\\tags.csv")
      .select("userId", "movieId", "tag")
      ).cache()

    //Dataframe com os ratings
    ratings_DF.show(10)

    //Dataframe com os titles
    movies_DF.show(10)

    //Número de linhas (instâncias) do Dataframe com titles
    movies_DF.count()

    //Dataframe com as tags
    tags_DF.show

    //Número de linhas (instâncias) do Dataframe com as tags
    tags_DF.count()

    //Enriquecer o Dataframe com ratings com as info dos
    //Dataframes com títulos e tags
    val ratings2_DF = (ratings_DF
      .hint("broadcast")
      .join(movies_DF, Seq("movieId"), "left")
      .join(tags_DF, Seq("userId", "movieId"), "left")
      .orderBy("userId", "movieId")
      )

    //Dataframe de ratings enriquecido
    ratings2_DF.show(10)

    //Número de linhas (instâncias) do Dataframe
    //enriquecido com titles e tags
    ratings2_DF.count()

    //Análise de Dados e Data Visualization
    val ratings3_DF = (ratings2_DF
      .withColumn("title", trim(col("title")))
      .withColumn("genres2", explode(split(col("genres"), "[|]")))
      )
      ratings3_DF.show()

    //Número de filmes assistidos e média
    //dos ratings por gênero (genre)
    val genres_DF = (ratings3_DF
      .groupBy("genres2")
      .agg(
        count("movieId").alias("movieId_count"),
        round(avg("rating"), 1).alias("rating_avg"),
        min("rating").alias("rating_min"),
        max("rating").alias("rating_max"),
        round(stddev("rating"), 1).alias("rating_std")
      )
      .orderBy(desc("movieId_count"))
      )
    genres_DF.show(10)

    //Número de filmes assistidos e média
    //dos ratings por filme (title)
    val titles_DF = (ratings2_DF
      .groupBy("title")
      .agg(
        count("userId").alias("userId_count"),
        round(avg("rating"), 1).alias("rating_avg"),
        min("rating").alias("rating_min"),
        max("rating").alias("rating_max"),
        round(stddev("rating"), 1).alias("rating_std")
      )
      .where(col("userId_count") > 10000)
      .orderBy(desc("userId_count"))
      )
    titles_DF.show(10)

    //Número de filmes assistidos e média
    //dos ratings por usuário (userId)
    val users_DF = (ratings2_DF
      .groupBy("userId")
      .agg(
        count("movieId").alias("movieId_count"),
        round(avg("rating"), 1).alias("rating_avg"),
        min("rating").alias("rating_min"),
        max("rating").alias("rating_max"),
        round(stddev("rating"), 1).alias("rating_std")
      )
      .orderBy(desc("movieId_count"))
      )
    users_DF.show(10)

    ss.stop()
  }
}
