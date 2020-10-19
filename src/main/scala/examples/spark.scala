package examples
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object spark {
  def conn(conf: SparkConf): SparkSession ={
    val spark = SparkSession.builder()
      .appName("Poc BigDL")
      .config("spark.executor.cores",1)
      .config("spark.cores.max",2)
      .config("spark.jars","/home/gervasio/Documents/scala/datascienceacademy/Projetos/poc-bigdl/out/artifacts/poc_bigdl_jar/poc-bigdl.jar")
      .config(conf)
//      .config("spark.driver.memory","2g")
//      .config("spark.work.memory","2g")
//      .master("local")
      .master("spark://sparkmaster.localdomain:7077")
      .getOrCreate()
    return spark
  }
}
