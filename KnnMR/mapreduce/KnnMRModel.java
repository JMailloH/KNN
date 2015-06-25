package org.apache.mahout.classifier.KnnMR.mapreduce;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
//import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.classifier.KnnMR.data.Data;
import org.apache.mahout.classifier.KnnMR.data.DataConverter;
//import org.apache.mahout.classifier.basic.data.DataConverter;
import org.apache.mahout.classifier.KnnMR.data.DataLoader;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.classifier.KnnMR.mapreduce.partial.PartialBuilder;
import org.apache.mahout.classifier.KnnMR.utils.KnnMRUtils;
import org.apache.mahout.classifier.KnnMR.builder.IBLclassifier;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.*;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.Distance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.KNN;
import org.apache.mahout.keel.Dataset.InstanceSet;
//import org.apache.mahout.keel.Dataset.Instance;
import org.apache.mahout.classifier.KnnMR.data.Instance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import com.google.common.io.Closeables;

public class KnnMRModel extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(KnnMRModel.class);
  
  private Path dataPath;
  private Path datasetPath;
  private Path headerPath;
  private Path testPath;
  private Path outputPath;
  //private Path timePath;  
  private String dataName;
  private String kneighbors;
  private String classifier="KNN";
  private String reduce="OPC1";
  private String testName;
  private long time;
 
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
	// Primero lectura de parámetros y control de que no falte:
	
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
        .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
        .withDescription("The path of the file descriptor of the dataset").create();

    Option header = obuilder.withLongName("header").withShortName("he").withRequired(true)
        .withArgument(abuilder.withName("header").withMinimum(1).withMaximum(1).create())
        .withDescription("Header of the dataset in Keel format").create();

    Option testOpt = obuilder.withLongName("testset").withShortName("ts").withRequired(true)
            .withArgument(abuilder.withName("testset").withMinimum(1).withMaximum(1).create())
            .withDescription("The path of the file descriptor of the testset").create();

    Option classifier = obuilder.withLongName("classifier").withShortName("cl").withRequired(false)
            .withArgument(abuilder.withName("classifier").withMinimum(1).withMaximum(1).create())
            .withDescription("Classifier: KNN or ... Default: KNN").create();

    Option reduce = obuilder.withLongName("reduce").withShortName("rd").withRequired(false)
            .withArgument(abuilder.withName("reduce").withMinimum(1).withMaximum(1).create())
            .withDescription("Reduce: OPC1 or ... Default: OPC1").create();
    
    Option kNeighbors = obuilder.withLongName("kNeighbors").withShortName("kn").withRequired(true)
            .withArgument(abuilder.withName("kNeighbors").withMinimum(1).withMaximum(1).create())
            .withDescription("K nearest neighbors: Number. Default: 1").create();
    
    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true)
        .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
        .withDescription("Output path, will contain the preprocessed dataset").create();
    
    Option helpOpt = obuilder.withLongName("help").withShortName("h")
        .withDescription("Print out help").create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt).withOption(header).withOption(testOpt).withOption(classifier).withOption(reduce).withOption(kNeighbors).withOption(outputOpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return -1;
      }
      
      // Obtenemos los parámetros que nos interesen:
      dataName = cmdLine.getValue(dataOpt).toString();
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      testName = cmdLine.getValue(testOpt).toString();
      String outputName = cmdLine.getValue(outputOpt).toString();
      String headerName = cmdLine.getValue(header).toString();
      String kNeighborsName = cmdLine.getValue(kNeighbors).toString();
      
      
      if (cmdLine.hasOption(classifier))
    	  this.classifier= cmdLine.getValue(classifier).toString();
      
      if (cmdLine.hasOption(reduce))
    	  this.reduce= cmdLine.getValue(reduce).toString();

      if (log.isDebugEnabled()) {
        log.debug("data : {}", dataName);
        log.debug("dataset : {}", datasetName);
        log.debug("header : {}", header);
        log.debug("test : {}", testName);
        log.debug("kNeighbors : {}", kNeighborsName);
        log.debug("classifier : {}", classifier);
        log.debug("reduce : {}", reduce);
        log.debug("output : {}", outputName);

      }

      dataPath = new Path(dataName);
      datasetPath = new Path(datasetName);
      testPath = new Path(testName);
      outputPath = new Path(outputName);
      headerPath = new Path(headerName);
      kneighbors = kNeighborsName;

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
      return -1;
    }
   
    buildModel();
    
    return 0;
  }
  

  private void buildModel() throws IOException, ClassNotFoundException, InterruptedException {
    // make sure the output path does not exist
    FileSystem ofs = outputPath.getFileSystem(getConf());

    if (ofs.exists(outputPath)) {
      log.error("Output path already exists");
      return;
    }

    IBLclassifier classifier = new IBLclassifier(this.classifier);
        
    FileSystem hfs = headerPath.getFileSystem(getConf());
    InstanceSet cabecera= KnnMRUtils.readHeader(hfs, this.headerPath);
    
    // Here We refer to the mapper/reducer classes, establish the corresponding key and value classes.
    
    log.info("KnnMR: Partial Mapred implementation"); 
    log.info("KnnMR: Preprocessing the dataset...");
    
    
    Builder modelBuilder = new PartialBuilder(classifier, dataPath, datasetPath, getConf(), cabecera.getHeader(), Integer.parseInt(kneighbors), outputPath.toString(), testName, reduce);
    
    System.out.println("DataName: "+dataName);
    System.out.println("datasetName: "+datasetPath);
    System.out.println("outputName: "+outputPath);
    System.out.println("TestSetName: "+testPath);
    System.out.println("headerName: "+headerPath);
    System.out.println("kNeighborsName: "+kneighbors);
    System.out.println("Classifier: "+classifier.classifier);
    
    time = System.currentTimeMillis();
    MapredOutput output=modelBuilder.build();
    //modelBuilder.build();
    time = System.currentTimeMillis() - time;
    log.info("KnnMR: Build Time: {}", KnnMRUtils.elapsedTime(time));
    log.info("KnnMR: Build Time in seconds: {}", KnnMRUtils.elapsedSeconds(time));
	

    //Divide the output.
    ArrayList<int[]> resultingSet = output.getOut();
    ArrayList<Long> mapTimes = output.getTimer();
    long reduceTime = output.getTime();
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    Map<String, Integer> parser = new HashMap<String, Integer>();

	//Hacemos nuestra propia conversión del label (String) a entero. Escribiremos resultado con esta conversión para poderla releer en KnnMRModel.java
	//Separamos la linea que contiene las etiquetas de la clase.
	String atriClass = null;
	
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
	//Creamos y rellenamos la matriz de confusión.
	
	//Leemos desde la cabecera el número de clases que hay
	FileSystem fs = headerPath.getFileSystem(getConf());
	
	FSDataInputStream input = fs.open(headerPath);
    Scanner scanner = new Scanner(input, "UTF-8");
	int num_classes = 0;
    while (scanner.hasNextLine()) {
        String line = scanner.nextLine();
        if((line.indexOf ("@attribute class")!= -1) || (line.indexOf ("@attribute Class")!= -1) ){
            num_classes = 1;
            for(int j = line.length() ; j > 0 ; j--){
          	  if(line.substring(j-1,j).equals(",")){
          		  num_classes += 1;
          	  }
            }
            
            
            if(line.indexOf ("@attribute class")!= -1){
            	atriClass = line.substring(line.indexOf ("@attribute class")+16);
        	}else if(line.indexOf ("@attribute Class")!= -1){
            	atriClass = line.substring(line.indexOf ("@attribute Class")+16);
        	}
        	//atriClass = atriClass.substring(0,atriClass.indexOf ("@"));
        	
            //System.out.println(atriClass);
            
        	//Limpiaos de corchetes y espacio
        	atriClass = atriClass.replace(" ", "");
        	atriClass = atriClass.replace("{", "");
        	atriClass = atriClass.replace("}", "");
        	
        	String[] labels;
        	labels = atriClass.split(",");
        	
        	//System.out.println("Etiquetas extraidas de la cabecera: ");
        	
        	for(int i = 0 ; i < labels.length ; i++){
        		//System.out.println(labels[i]);
        		parser.put(labels[i], i);
        	}
            
            
        }
    }
    
    System.out.println("Número de clases: " + num_classes);
    
	scanner.close();

	int [][] confusionMatrix = new int[num_classes][num_classes];
    for(int x = 0 ; x < confusionMatrix.length ; x++){
    	for(int y = 0 ; y < confusionMatrix[0].length ; y++){
    		confusionMatrix[x][y] = 0;
    	}
    }
    
    //Doing structure right-class predicted-class.
    ArrayList<int []> rightPredictedClass = new ArrayList<int []>();

    String[] lineSplit;
    Path testPath = new Path(this.testName);
	//Leemos el conjunto de test linea a linea
	FileSystem fsTest = testPath.getFileSystem(getConf());
	FSDataInputStream inputTest = fsTest.open(testPath);
    Scanner scannerTest = new Scanner(inputTest, "UTF-8");
    boolean primero = true;
    System.out.println(testPath);
    while (scannerTest.hasNextLine()) {
    	//context.progress();
        String line = scannerTest.nextLine();
        lineSplit = line.split(",");
		int[] aux = new int[2];
		aux[0] = parser.get(lineSplit[lineSplit.length-1]);
    	aux[1] = 0;
    	rightPredictedClass.add(aux);
    }
    scanner.close();
    
    for(int i = 0 ; i < resultingSet.size() ; i++){
    	int[] aux = new int[2];
		aux[0] = rightPredictedClass.get(resultingSet.get(i)[0])[0];
    	aux[1] = resultingSet.get(i)[1];
    	rightPredictedClass.set(resultingSet.get(i)[0],aux);
    }        	
        	
    //Predictions.txt -> primera columna clase predicha. segunda columna clase real.	
	FileSystem outFS = outputPath.getFileSystem(getConf());
	Path filenamePath = new Path(outputPath, "Predictions").suffix(".txt");
	FSDataOutputStream ofile = outFS.create(filenamePath);
	StringBuilder in = new StringBuilder();
	String inString = null;
	in.append("***Predictions.txt ==> 1th column predicted class; 2on column right class***").append('\n');
	
	for(int i = 0 ; i < rightPredictedClass.size() ; i++){
	    confusionMatrix[rightPredictedClass.get(i)[1]][rightPredictedClass.get(i)[0]] = confusionMatrix[rightPredictedClass.get(i)[1]][rightPredictedClass.get(i)[0]]+1;
		in.append(rightPredictedClass.get(i)[0]+"\t"+rightPredictedClass.get(i)[1]+"\n");
	}
    
	inString = in.toString();
	ofile.writeBytes(inString);
	ofile.close();	
	scanner.close();
	

	
	//Results.txt ==> Contain: Accuracy; Confusion Matrix; Time of the run
	filenamePath = new Path(outputPath, "Results").suffix(".txt");
	ofile = outFS.create(filenamePath);
	in = new StringBuilder();
	in.append("***Results.txt ==> Contain: Confusion Matrix; Accuracy; Time of the run***\n").append('\n');
	
	//Write the confusion matrix on String and calulate the accuracy 
	float rightPredicted = 0;
	float wrongPredicted = 0;
    for (int x = 0 ; x < num_classes ; x++){
    	for(int y = 0 ; y < num_classes ; y++){
    		if(x==y){
    			rightPredicted += confusionMatrix[x][y];
    		}else{
    			wrongPredicted += confusionMatrix[x][y];
    		}
    		in.append(confusionMatrix[x][y]+"\t");
    	}
    	in.append('\n');
    }
	in.append('\n');

	//Write the accuracy
	in.append(rightPredicted/(wrongPredicted+rightPredicted)+"\n\n");
	
	//Write the time of run on String
	in.append(KnnMRUtils.elapsedSeconds(time)+"\n\n");
    
	//Write all on hdfs
	inString = in.toString();
	ofile.writeBytes(inString);
	ofile.close();	

	
	
	
	//Times.txt ==> Contain: run maps time; run reduce time; run clean up reduce time; Time of complete the run
	filenamePath = new Path(outputPath, "Times").suffix(".txt");
	ofile = outFS.create(filenamePath);
	in = new StringBuilder();
	in.append("***Times.txt ==> Contain: run maps time; run reduce time; run clean up reduce time; Time of complete the run***\n").append('\n');
	
	//Write the map time on String
	in.append("@mapTime\n");
	for(int i = 0 ; i < mapTimes.size() ; i++){
		in.append(KnnMRUtils.elapsedSeconds2(mapTimes.get(i))).append('\n');
	}
	
	//Write the clean up reduce time on String
	in.append("\n@reduceTime\n");
	in.append(KnnMRUtils.elapsedSeconds2(reduceTime)).append('\n');
	
	//Write the time of run on String
	in.append("\n@totalRunTime\n");
	in.append(KnnMRUtils.elapsedSeconds2(time)+"\n\n");
    
	//Write all on hdfs
	inString = in.toString();
	ofile.writeBytes(inString);
	ofile.close();	

  }
  
  protected static Data loadData(Configuration conf, Path dataPath, Dataset dataset) throws IOException {
    log.info("KnnMR: Loading the data...");
    FileSystem fs = dataPath.getFileSystem(conf);
    Data data = DataLoader.loadData(dataset, fs, dataPath);
    log.info("KnnMR: Data Loaded");
    
    return data;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new KnnMRModel(), args);
  }
  
}
