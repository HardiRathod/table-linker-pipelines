

1. Download Data From NIH Reporter and Convert to KGTK files

'''
        Input:  https://exporter.nih.gov/ExPORTER_Catalog.aspx
	
	Run this notebook to convert data from NIH Reporter csv files to KGTK files
	
	1_NIH_Reporter_to_KGTK_files.ipynb
	
	Results: nih_kgtk_files 
'''
	
	
2.  Converting Raw KGTK Files to Table Linker / Tableau Files

	Input: nih_kgtk_files
	
	Run 2_KGTK_files_to_tl_and_tableau.ipynb notebook
	
	Final outputs are:
	
		Files for table-linker Table_Linker_Complete_Files 



3. Run table-linker for organization and people:

	Input:Table_Linker_Complete_Files 
	
	Run: Table_linker_pipelines 
	
	Results: person_with_qid, organization_with_qid



4. Converting Raw KGTK Files to Table Linker / Tableau Files

	Input: 
		Wikidata Files (steps 1-4) in Build ElasticSearch Index Files for Table Linker, 
		nih_kgtk_files and nih_project_files
		person_with_qid, organization_with_qid
		Co-Investigator file, coinvestigators.compact.tsv
		
	Run 3 tableau.ipynb notebook
	
	Final outputs are:
		An excel file for tableau:  nih_0909_new3.xlsx



Visualizating Output File in Tableau 
	See Video: https://youtu.be/XZGLJhLFZQ0
