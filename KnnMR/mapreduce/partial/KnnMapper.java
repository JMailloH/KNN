package org.apache.mahout.classifier.KnnMR.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.KnnMR.mapreduce.MapredOutput;
import org.apache.mahout.classifier.KnnMR.mapreduce.Builder;
import org.apache.mahout.classifier.KnnMR.mapreduce.MapredMapper;
import org.apache.mahout.classifier.KnnMR.utils.KnnMRUtils;
import org.apache.mahout.classifier.KnnMR.data.Data;
import org.apache.mahout.classifier.KnnMR.data.DataConverter;
import org.apache.mahout.classifier.KnnMR.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Distance;
import org.apache.mahout.classifier.KnnMR.utils.Pair;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class KnnMapper extends MapredMapper<LongWritable,Text,StrataID,MapredOutput>{
  
  private static final Logger log = LoggerFactory.getLogger(KnnMapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
  /** mapper's partition */
  private int partition;
  
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
  
  public int getFirstTreeId() {
    return firstId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    context.progress();
    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf), Builder.getHeader(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
   */
  protected void configure(int partition, int numMapTasks, String header) {
    converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.partition = partition;
    this.header=header;
    log.debug("partition : {}", partition);
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));
    context.progress();
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
	//Taking the time
	long time = System.currentTimeMillis();

    // prepare the data
	  
    log.debug("partition: {} numInstances: {}", partition, instances.size());
        
    context.progress();

    Data data = new Data(getDataset(), instances);
    
    log.info("KnnMR: Size of data partition= "+data.size());
    
    log.info("cabecera: "+header);
    
    context.progress();
    KnnMRUtils.readHeader(this.header);

    try {
		classifier.build(data, context);
	} catch (Exception e) {
		// TODO Bloque catch generado automáticamente
		e.printStackTrace();
	}
    
    //Read the training set
    PrototypeSet training = new PrototypeSet(data);
    context.progress();
 
    //Calculamos los k vecinos para cada instancia del conjunto de test.
    
   // Pair <Double, double> ClassDistance;
   ArrayList<Pair <Double[], Double[]>> resultados = new ArrayList<Pair <Double[], Double[]>>();

//    ArrayList<PrototypeSet> resultados = new ArrayList<PrototypeSet>();
   int itert = 0;

    if(this.classifier.classifier.equals("KNN")){
    	
    	Path testPath = new Path(this.testName);
    	//Leemos el conjunto de test linea a linea
    	FileSystem fs = testPath.getFileSystem(context.getConfiguration());
    	
    	FSDataInputStream input = fs.open(testPath);
	    Scanner scanner = new Scanner(input, "UTF-8");
        boolean primero = true;
        while (scanner.hasNextLine()) {
	    	context.progress();
	        String line = scanner.nextLine();

	        Instance nueva = converter.convert(line);
	        String[] lineSplit;
	        lineSplit = line.split(",");
	        System.out.println("Clase leida del fichero: " + lineSplit[lineSplit.length-1]);
	       // org.apache.mahout.keel.Dataset.Instance currentInstance = new org.apache.mahout.keel.Dataset.Instance( line, true, 1);
    		//Prototype current= new Prototype(currentInstance);
	        Prototype current= new Prototype(nueva.get()); 
	        
	        System.out.println("Clase con el convert: " + current.getOutput(0));
	        System.out.println("A ver si le doy la vuelta: " + converter.getLabelClass(current.numberOfInputs(), current.getOutput(0)));
	        
	        PrototypeSet vecinos = KNN.getNearestNeighbors(current, training, this.Kneighbour);
	        
	        Double[] distances = new Double[this.Kneighbour];
	        Double[] classes = new Double[this.Kneighbour];

	        double auxClass = 0;
	        for(int i=0; i< this.Kneighbour; i++){
	        	
	        	distances[i] = Distance.squaredEuclideanDistance(vecinos.get(i), current);
	        	auxClass = Double.parseDouble(converter.getLabelClass(current.numberOfInputs(), current.getOutput(0)));
	        	
	        	//Clase con la transformación deshecha NO FUNCIONA POR LOS LABELS DE LAS CLASES
	        	//classes[i] = auxClass;
	        	
	        	//Clase con la transformación
	        	classes[i] = vecinos.get(i).getOutput(0);       
	        	
	        	//System.out.println(distances[i]+", "+classes[i]);
	        	
	        }
	        
	        //System.out.println("*******");
	        
	        // Classes, then - distances
	        Pair <Double[], Double[]> result= new Pair <Double[], Double[]>(classes, distances);
	        
	        resultados.add(result);
	        
	        StrataID key = new StrataID();
	        key.set(0, itert);
	        MapredOutput emOut = new MapredOutput(resultados,1,time);  
	        context.write(key, emOut);
	        
	        itert++;
	    	//System.out.println("Instancia: " + Integer.parseInt(key.toString()) + " | " + result.first()[0] + " " + result.second()[0]);

	        resultados.clear();

	        
    	}
		scanner.close();
    }
    time = System.currentTimeMillis() - time;
    
    /*StrataID key = new StrataID();

    key.set(partition, firstId + 1);
    
   
    MapredOutput emOut = new MapredOutput(resultados,1,time);  

    context.write(key, emOut);
	System.out.println("Fin del map");*/

    //}
   
  }
}
