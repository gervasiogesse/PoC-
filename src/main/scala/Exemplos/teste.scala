package Exemplos
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils._
import org.tensorflow.example.Example

import spark._


object teste {
  def main(args: Array[String]): Unit = {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = Engine.createSparkConf()
      .set("spark.task.maxFailures", "1")
    val spark = conn(conf)
    Engine.init
    println("BigDL"+BIGDL_VERSION)
    val linear = Linear.apply(3, 5)
    //Input layer
    def randomn(): Double = RandomGenerator.RNG.uniform(0, 1)
    val input = Tensor(2, 3)
    input.apply1(x => randomn().toFloat)
    println("input:")
    println(input)
    val module = Input
    println("output sem mudanca:")
    println(module.apply().element.forward(input))
    //Echo layer
    val echo = Echo
    val output = echo.apply().forward(input)
    println(output)
  }
}
