package org.apache.mahout.classifier.KnnMR.mapreduce.partial;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.KnnMR.mapreduce.MapredOutput;
import org.apache.mahout.classifier.KnnMR.mapreduce.Builder;
import org.apache.mahout.classifier.KnnMR.mapreduce.MapredReducer;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.classifier.KnnMR.data.Data;
import org.apache.mahout.classifier.KnnMR.data.DataConverter;
import org.apache.mahout.classifier.KnnMR.data.Instance;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.Prototype;
import org.apache.mahout.keel.Algorithms.Instance_Generation.utilities.KNN.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class KnnMRReducer extends MapredReducer<StrataID,MapredOutput,StrataID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(KnnMRReducer.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;
  
  /**first id */
  private int firstId = 0;
  
 
  /** will contain all instances if this mapper's split */
  private final List<Instance> instances = Lists.newArrayList();
   

  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    log.info("Configuring reducer");

    configure(conf.getInt("mapred.task.partition", -1), Builder.getNumMaps(conf));
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
  protected void configure(int partition, int numMapTasks) {
   // converter = new DataConverter(getDataset());

    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID");
    this.strata = partition;
    
    log.debug("partition : {}", partition);
  }
   
 // KEYIN id, Iterable<VALUEIN> rs, 
// aquí no funciona con los tipos de datos concretos, uso abstractos en MapredReducer
  /*
  protected void reduce(StrataID id, MapredOutput rs, Context context) throws IOException, InterruptedException {
	  log.info("añadiendo instancias "+rs.getRS().size());
	  join.add(rs.getRS());
	  
	//  context.write(id, new PrototypeSet(rs.getRS()));
	  //instances.add(converter.convert(value.toString()));
  }
  */

  // para hacerlo bien, esta fase hay que hacerla en el principal, no en el mapper.

  
  protected void cleanup(Context context) throws IOException, InterruptedException {
	 // log.debug("partition: {} numInstances: {}", partition, instances.size());
    
	    log.info("reduce: {} numInstances: {}", strata, join.size());
	   	   
	    StrataID key = new StrataID();

	    key.set(strata, firstId + 1);
	      
	    
	    //join.print();
	    context.progress();
	    
	    join=new PrototypeSet(ENN(join));
	    
	    log.info("Tamaño despues de la limpieza: {} ", join.size());

	    if (!isNoOutput()) {
	    	MapredOutput emOut = new MapredOutput(1);
	        context.write(key, emOut);
	    }
	    
	    
	    //save it in disk
	    join.save("prueba.dat");

  }
  
 
  /**
   * 
   * Edited nearest neighbor of T. , PrototypeSet labeled
   * @return
   */
  protected PrototypeSet ENN (PrototypeSet T)
  {
	//T.print();
	 PrototypeSet Sew = new PrototypeSet (T);
	
	 //this.k = 7;
	  // Elimination rule kohonen
	  int majority = 3/2 + 1;
	 // System.out.println("Mayor�a " + majority);


	int toClean[] = new int [T.size()];
	Arrays.fill(toClean, 0);
	int pos = 0;
	  
	for ( Prototype p : T){
		 double class_p = p.getOutput(0);
		 PrototypeSet neighbors = KNN.knn(p, T, 3); // labeled
		
		  int counter= 0;
		  for(Prototype q1 :neighbors ){
			double class_q1 = q1.getOutput(0);
			
			if(class_q1 == class_p){
				counter++;
			} 
			
		  }
		  
		  //System.out.println("Misma clase = "+ counter);
		  if ( counter < majority){ // We must eliminate this prototype.
			  toClean [pos] = 1; // we will clean			  
		  }
		   pos++;
	}
	
	//Clean the prototypes.
	PrototypeSet aux= new PrototypeSet();
	for(int i= 0; i< toClean.length;i++){
		if(toClean[i] == 0)
			aux.add(T.get(i));
		
	}
	//Remove aux prototype set
	
	Sew = aux;
	
	//System.out.println("Result of filtering");	
	//Sew.print();

	return Sew;
	  
  }
  
  
}
