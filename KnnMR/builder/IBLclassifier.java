package org.apache.mahout.classifier.KnnMR.builder;

import org.apache.hadoop.mapreduce.Mapper.Context;

import org.apache.mahout.classifier.KnnMR.data.Data;
import org.apache.mahout.classifier.KnnMR.data.Dataset;
import org.apache.mahout.keel.Algorithms.Instance_Generation.Basic.PrototypeSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IBLclassifier  {
  
  private static final Logger log = LoggerFactory.getLogger(IBLclassifier.class);	
  int nClasses, nLabels;

  public String classifier = "KNN";
  
  public String header;
  //IPLDEGenerator algorithm;
  //HandlerSSMASFLSDE algorithm2;
  //IPLDE_windowingGenerator algorithmIPADEwin;
  
	//  strata[i].print();
	  
  public IBLclassifier() {
  }
  
  public IBLclassifier(String cla)
  {
	  this.classifier = cla;
  }
  
  public void setNLabels(int nLabels) {
    this.nLabels = nLabels;
  }


  public void setHeader(String header){
	  this.header=header;
  }
  
  public void build(Data data, Context context) throws Exception {
    //We do here the algorithm's operations

	Dataset dataset = data.getDataset();
	 
	nClasses = dataset.nblabels();
	
	//Gets the number of input attributes of the data-set
	int nInputs = dataset.nbAttributes() - 1;
	
	//It returns the class labels
	String clases[] = dataset.labels();
	
	// data has the instance + label
	//for(int i=0; i<= nInputs; i++)
	//	log.info("prueba: "+data.get(0).get()[i]);
	context.progress();

	if(this.classifier.equalsIgnoreCase("KNN")){
		log.info("Clasificador: ejecutando KNN...");
	}
		//algorithm = new IPLDEGenerator(context, data, 1, 10000, 8, 20, 0.5, 0.9, 0.03, 0.07);
	/*}else if(this.PGmethod.equalsIgnoreCase("SSMASFLSDE")){
		log.info("PGgenerator: ejecutando SSMASFLSDE...");
		algorithm2 = new HandlerSSMASFLSDE();
		algorithm2.ejecutar(data, context);
	}*/
	/*else if(this.PGmethod.equalsIgnoreCase("IPADE_windowing")){	
		log.info("PGgenerator: ejecutando IPADE con windowing...");
	//	algorithmIPADEwin = new IPLDE_windowingGenerator(context, data, 1, 10000, 8, 20, 0.5, 0.9, 0.03, 0.07,5000);
	 * 
	 
	else{
		log.info("PGgenerator: No hay reducción, guardo el fichero de entrada tal cual.");
		algorithm2 = new HandlerSSMASFLSDE();
		algorithm2.reducedSet = new PrototypeSet(data, context);
	}
	 */
	
	log.info("Data size = "+data.size());

  }
  
  public PrototypeSet reduceSet() {
	  PrototypeSet output=null;
	  
	  if(this.classifier.equalsIgnoreCase("IPADE")){
		 //output=algorithm.reduceSet();
		 output.applyThresholds();

	  }/*else if(this.PGmethod.equalsIgnoreCase("IPADE_windowing")){
			 output=algorithmIPADEwin.reduceSet();
			 output.applyThresholds();
	  }*/
	  else{
		  //soutput=algorithm2.reducedSet;
	  }
	  
	  log.info("PG: RS size = "+output.size());
   
	  //log.info("\n"+output.toString());
	  return output;
  }


}
