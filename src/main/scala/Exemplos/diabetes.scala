package Exemplos

import breeze.linalg.SparseVector
import breeze.numerics.sqrt
import com.intel.analytics.bigdl.BIGDL_VERSION
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.dlframes._
import com.intel.analytics.bigdl.optim.Adam
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions._

// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StandardScaler

import spark._

/*
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
 */
object diabetes extends App {
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
  val data = spark.read.option("header","false")
    .option("inferSchema","true")
    .option("mode", "DROPMALFORMED")
    .format("csv")
    .load("file:///mnt/teste/pima-indians-diabetes.csv").na.drop()
  println(data.count())
  data.printSchema()
  data.groupBy("_c8").count().show()
//  # BigdL doesn't like 0 (zero) in label column
//  # so I am going to add +1 to label
  val cleanData = data.select((data("_c8").cast("double")+1).as("label"),
    data("_c0").cast("double").as("_c0"),
    data("_c1").cast("double").as("_c1"), data("_c2").cast("double").as("_c2"),
    data("_c3").cast("double").as("_c3"), data("_c4").cast("double").as("_c4"),
    data("_c5").cast("double").as("_c5"),
    data("_c6").cast("double").as("_c6"), data("_c7").cast("double").as("_c7")
  )
  //Não é mais necessário usar array a versao 0.10 já suorta o tipo DenseVector
//  val cleanData = data.select((data("_c8").cast("double")+1).as("label"),
//    array(data("_c0").cast("double").as("_c0"),
//      data("_c1").cast("double").as("_c1"), data("_c2").cast("double").as("_c2"),
//      data("_c3").cast("double").as("_c3"), data("_c4").cast("double").as("_c4"),
//      data("_c5").cast("double").as("_c5"),
//      data("_c6").cast("double").as("_c6"), data("_c7").cast("double").as("_c7")
//    ).as("features1")
//  )
  cleanData.show(2)
  cleanData.printSchema()
  println(cleanData.count())
//   Cria um novo objeto VectorAssembler chamado assembler as features
  // Defina a coluna de saída
  val assembler = new VectorAssembler()
    .setInputCols(Array("_c0","_c1","_c2","_c3","_c4","_c5","_c6","_c7"))
    .setOutputCol("features1")
  val fv1 = assembler.transform(cleanData).select($"features1",$"label")
  val scaler =  new StandardScaler().setInputCol("features1").setOutputCol("features")
  val fv = scaler.fit(fv1).transform(fv1)
  fv.show(truncate = false)
  fv.printSchema()
//   Use randomSplit para criar uma divisão em treino e teste em 70/30
  val Array(training, test) = fv.randomSplit(Array(0.7, 0.3), seed = 12345)
  val qtd = training.count().toInt
  println(qtd)
  // Definindo o numero de camadas ocultas
  val n_input = 8
  val n_classes = 2
  // Função para determinar a quantdade de neuronios da primeira camada para iniciar os testes
  val n_hidden_1 = sqrt(sqrt((n_classes + 2) * n_input) + 2 * sqrt(n_input /(n_classes+2))).toInt + 3
  println("n camadas 1 = "+n_hidden_1)
  // Segunda camada
  val n_hidden_2 = n_classes * sqrt(n_input / (n_classes + 2)).toInt
  println("n camadas 2 = "+n_hidden_2)
  // Incializa um container sequencial
  val nn = Sequential
    .apply()
    .add(Linear.apply(n_input,n_hidden_1))
    .add(Linear.apply(n_hidden_1,n_hidden_2))
    .add(Linear.apply(n_hidden_2,n_hidden_2))
//    .add(Linear.apply(n_hidden_2,n_hidden_2))
    .add(Linear.apply(n_hidden_2, n_classes))
    .add(LogSoftMax.apply())
  val criterion = ClassNLLCriterion()
  val estimator = new DLClassifier(model=nn, criterion=criterion,Array(n_input))
    .setMaxEpoch(100)
    .setBatchSize(526)
//    .setLearningRate(0.1)
    .setLabelCol("label").setFeaturesCol("features")
        .setOptimMethod(new Adam[Float](learningRate = 0.001))
  println("Inicio do treino")
  val model = estimator.fit(training)
  println("Fim do treino")
  val predictions = model.transform(test)
  predictions.groupBy("prediction").count().show()
  test.groupBy("label").count().show()
  predictions.show(truncate = false)
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
}
