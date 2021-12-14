

1. Download Data From NIH Reporter and Convert to KGTK files
Input:  https://exporter.nih.gov/ExPORTER_Catalog.aspx
Run this notebook to convert data from NIH Reporter csv files to KGTK files
1_NIH_Reporter_to_KGTK_files.ipynb
Results: nih_kgtk_files 
	
4.  Converting Raw KGTK Files to Table Linker / Tableau Files
Input: 
nih_kgtk_files
Run 2_KGTK_files_to_tl_and_tableau.ipynb notebook
Final outputs are:
Files for table-linker Table_Linker_Complete_Files 



2. Run table-linker for organization and people:

Input:Table_Linker_Complete_Files 
Run: Table_linker_pipelines 
Results: person_with_qid, organization_with_qid



4.  Converting Raw KGTK Files to Table Linker / Tableau Files
Input: 
Wikidata Files (steps 1-4) in Build ElasticSearch Index Files for Table Linker, 
nih_kgtk_files and nih_project_files
person_with_qid, organization_with_qid
Co-Investigator file, coinvestigators.compact.tsv
Run 3 tableau.ipynb notebook
Final outputs are:
An excel file for tableau:  nih_0909_new3.xlsx



Visualizating Output File in Tableau 

Input excel file (nih_0909_new3.xlsx) and setting the geographic role of lat, long, and FIPS (as county). Similar procedure to the animation below:


Add long to column and lat to row to create a map, and add more details if needed
Creating a sheet:
Drag longitude to columns and latitude to rows
Drag a name/id to details
Drag other attributes to color and details as needed
For adding filters and linking the filters to multiple sheets:
Select Filters > <filter field>
On the Filters shelf, right-click the field and select Apply to Worksheets > All Using This Data Source.
Add a dual axis to graph using lat, long generated from FIPS to add background for map

Detailed instruction at https://www.tableau.com/about/blog/2016/11/10-map-tips-tableau-62949		


