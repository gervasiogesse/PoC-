package examples

import breeze.numerics.sqrt
import com.intel.analytics.bigdl.BIGDL_VERSION
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dlframes._
import com.intel.analytics.bigdl.optim.Adam
import spark.conn

// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler

import examples.spark

object camadasOcultas extends App {
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
    .load("file:///mnt/teste/Iris.csv")
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
  fv.show()
  // Use randomSplit para criar uma divisão em treino e teste em 70/30
  val Array(training, test) = fv.randomSplit(Array(0.7, 0.3), seed = 12345)
  val qtd = training.count().toInt
  println(qtd)
  // Definindo o numero de camadas ocultas
  val n_input = 4
  val n_classes = 3
  // Função para determinar a quantdade de neuronios da primeira camada para iniciar os testes
  val n_hidden_1 = sqrt(sqrt((n_classes + 2) * n_input) + 2 * sqrt(n_input /(n_classes+2))).toInt + 1
  println("n camadas 1 = "+n_hidden_1)
  // Segunda camada
  val n_hidden_2 = n_classes * sqrt(n_input / (n_classes + 2))
  println("n camadas 2 = "+n_hidden_2)
  // Incializa um container sequencial
  val nn = Sequential
    .apply()
    .add(Linear.apply(n_input,n_hidden_1))
    .add(Linear.apply(n_hidden_1, n_classes))
    .add(LogSoftMax.apply())
  val criterion = ClassNLLCriterion()
  val estimator = new DLClassifier(model=nn, criterion=criterion,Array(n_input))
    .setMaxEpoch(10)
    .setBatchSize(qtd)
    .setLearningRate(0.1)
    .setLabelCol("label").setFeaturesCol("features")
//      .setOptimMethod(new Adam[Float](learningRate = 0.1))
  println("Inicio do treino")
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
