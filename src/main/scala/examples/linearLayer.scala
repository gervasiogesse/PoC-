package examples
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils._


import spark._


object linearLayer extends App {
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  Logger.getLogger("org").setLevel(Level.ERROR)

  val conf = Engine.createSparkConf()
    .set("spark.task.maxFailures", "1")
  val spark = conn(conf)
  Engine.init
  println("BigDL"+BIGDL_VERSION)

  val model = Sequential
  //Hidden layer with ReLu
  model.apply().add(Linear.apply(4, 4))
  model.apply().add(ReLU.apply())
  //Output layer
  model.apply().add(Linear.apply(4, 3))
  model.apply().add(LogSoftMax.apply())
}
