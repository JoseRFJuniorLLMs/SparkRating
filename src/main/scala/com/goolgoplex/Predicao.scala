package com.goolgoplex

import breeze.numerics.{abs, round}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

/**
 * Machine Learning - Sistema de Recomendações de filmes
 * fonte de dados : https://grouplens.org/datasets/movielens/
 * @author web2ajax@gmail.com
 *
 */
object Predicao extends Serializable {

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

    //Definir as amostras de treinamento e validação
    //do modelo preditivo de recomendação
    val train_size = 0.75
    val test_size = 1 - train_size
    val Array(train, test) = ratings_DF.randomSplit(Array(train_size, test_size))
    test.count()

    //Set up do modelo de recomendação
    val als = (new ALS()
      .setMaxIter(15)
      .setAlpha(1.00)
      .setSeed(20111974)
      .setImplicitPrefs(false)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setNonnegative(true)
      .setItemCol("movieId")
      .setRatingCol("rating")
      )

    //Treinar o modelo de recomendação
    val model = als.fit(train)

    //Predições do modelo de recomendação (usando amostra de validação)
    model.setColdStartStrategy("drop")
    val predictions_test_DF = (model
      .transform(test)
      .withColumn("error", round(abs(col("rating") - col("prediction")), 1))
      .withColumn("prediction", round(col("prediction"), 1))
      .orderBy("movieId", "rating")
      ).na.drop()

    //Predições do modelo de recomendação (usando amostra de treinamento)
    val predictions_train_DF = (model
      .transform(train)
      .withColumn("error", round(abs(col("rating") - col("prediction")), 1))
      .withColumn("prediction", round(col("prediction"), 1))
      .orderBy("movieId", "rating")
      ).na.drop()

    //Carregar informações para enriquecimento do Dataframe com as predições
    val predictions2_test_DF = (predictions_test_DF
      .join(movies_DF, Seq("movieId"))
      .join(tags_DF, Seq("userId", "movieId"))
      )
    predictions2_test_DF.show()

    //Dataframe temporário com estatísticas sobre os usuários
    val users2_DF = (predictions2_DF
      .groupBy("userId")
      .agg(
        count("movieId").alias("movieId_count"),
        round(avg("rating"), 1).alias("rating_avg"),
        round(avg("prediction"), 1).alias("prediction_avg"),
        round(avg("error"), 1).alias("error_avg"),
        round(stddev("error"), 1).alias("error_std"),
        round(stddev("rating"), 1).alias("rating_std"),
        round(stddev("prediction"), 1).alias("prediction_std")
      )
      .orderBy(desc("movieId_count"))
      )

    //Dataframe temporário com estatísticas sobre os titles
    val titles2_DF = (predictions2_DF
      .groupBy("title")
      .agg(
        count("title").alias("title_count"),
        round(avg("rating"), 1).alias("rating_avg"),
        round(avg("prediction"), 1).alias("prediction_avg"),
        round(avg("error"), 1).alias("error_avg"),
        round(stddev("error"), 1).alias("error_std"),
        round(stddev("rating"), 1).alias("rating_std"),
        round(stddev("prediction"), 1).alias("prediction_std")
      )
      .orderBy(desc("title_count"))
      )

    //Calcular o RMSE
    val evaluator = (new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      )
    val rmse = evaluator.evaluate(predictions2_DF)
    println(s"Root-mean-square error = $rmse")

    //Top 10 filmes recomendados para cada usuário
    val userRecs_DF = model.recommendForAllUsers(10).orderBy("userId")

    //Top 10 usuários para cada filme
    val movieRecs_DF = model.recommendForAllItems(10).orderBy("movieId")

    //Salvar o Dataframe com as predições do modelo de recomendação
    predictions2_DF.write.mode("overwrite").json("/FileStore/tables/MovieLens_Prediction.json")

    //Top 10 recomendações para cada usuário (Dataframe construído anteriormente...)
    val userRecs_top10_DF = userRecs_DF.withColumn("userId", userRecs_DF("userId").cast("string")).select("userId", "recommendations." + "MovieId")
    userRecs_top10_DF.show(10)

  }
}