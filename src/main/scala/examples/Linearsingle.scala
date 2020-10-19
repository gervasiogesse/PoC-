package examples
import com.intel.analytics.bigdl.BIGDL_VERSION
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dlframes._

// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.VectorAssembler

import spark._

//Linear Perceptron

object Linearsingle extends App {
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  Logger.getLogger("org").setLevel(Level.ERROR)

  val conf = Engine.createSparkConf()
    .setAppName("Linear Single")
//    .setMaster("local[*]")
    .set("spark.task.maxFailures", "1")
  val spark = conn(conf)
  Engine.init
  println("BigDL"+BIGDL_VERSION)

  import spark.implicits._

  // Carregando os dados
  val data = spark.read.option("header","true")
    .option("inferSchema","true")
    .option("mode", "DROPMALFORMED")
    .format("csv")
    .load("file:///mnt/share/iris.csv")
  println(data.count())
  val cleanData = data.na.drop().select(data("Species").as("label").cast("double"),
    $"SepalLengthCm",$"SepalWidthCm",$"PetalLengthCm",$"PetalWidthCm")
  println(cleanData.count())
  cleanData.printSchema()
  cleanData.show(2)
  // Cria um novo objeto VectorAssembler chamado assembler as features
  // Defina a coluna de saída
  val assembler = new VectorAssembler()
    .setInputCols(Array("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"))
    .setOutputCol("features")
//  Use 'StandardScaler' to scale the features
  val fv = assembler.transform(cleanData).select($"features",$"label")
  fv.show(2)
  // Use randomSplit para criar uma divisão em treino e teste em 70/30
  val Array(training, test) = fv.randomSplit(Array(0.6, 0.4), seed = 12345)
  val qtd = training.count().toInt
  println(qtd)
  // Incializa um container sequencial
  val nn = Sequential
    .apply()
    .add(Linear.apply(4,3)) //Adiciona uma camada linear com n_input, n_output
    // n_input não deve ser inferior a quantidade de colunas do features
    // n_output não deve ser inferior a ndistict do label
//    .add(Linear.apply(2,2))
    .add(LogSoftMax.apply())

  val criterion = ClassNLLCriterion()
  val estimator = new DLClassifier(model=nn, criterion=criterion,Array(4))
    .setMaxEpoch(50)
    .setBatchSize(qtd)
    .setLearningRate(0.1)
    .setLabelCol("label").setFeaturesCol("features")
  println("Inicio do treino")
  val dataT = spark.sparkContext.parallelize(Seq(
    (Array(0.0, 1.0), 1.0),
    (Array(1.0, 0.0), 2.0),
    (Array(0.0, 1.0), 1.0),
    (Array(1.0, 0.0), 2.0)))
  val df = dataT.toDF("features", "label").repartition(2)
//  df.printSchema()
  // Necessário compiar as bibliotecas para o spark
  // bigdl-SPARK_2.4-0.10.0-jar-with-dependencies.jar para /opt/spark/jars
  val model = estimator.fit(training)
  println("Fim do treino")
  val predictions = model.transform(test)
  predictions.groupBy("prediction").count().show()
  test.groupBy("label").count().show()
  predictions.show()
  // Para métricas e avaliação, importe MulticlassMetrics
  import org.apache.spark.mllib.evaluation.MulticlassMetrics

  // Converta os resultados do teste em um RDD usando .as e .rdd
  val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd

  // Instanciar um novo objeto MulticlassMetrics
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)
  println("Acurácia:")
  println(metrics.accuracy)
  //Iris-setosa - 1, Iris-versicolor - 2, Iris-virginica - 3
}
