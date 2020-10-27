################################################################################
#######################>>>>>>>NOTES TO GROUP<<<<<<<<<###########################
################################################################################

# Note to fellow group members! I think we can differentiate tasks through the
# table of contents. The next code To-Do is being handled by Jacob, which is
# indicated in section 2 of the table of contents

''' Comment: In my opinion the paper is structured like the following: 
	
	(## no need to replicate ##)
    0) Topic modelling: Preprocessing, LDA, Gibbs sampler

	(### no need to replicate ###)
    1) Argument for relevance of within variation through apllication of model
       to existing models from literature
		1.1 Rainfall: Miguel & Satyanath (2011)
		1.2 External economic shocks and political constrains: Besley & Presson(2011b)
		1.3 Political institution dummies: Goldstone et al. (2010)
		1.4 ICEWS conflict events: Ward et al. (2013)
		1.5 Keyword counts: Chadefaux (2014)
		1.6 Country fixed effects only

    2) Topic model

    	(### very important to REPLICATE ###)
		2.1 Performance of topic model
		    2.1.1 run fixed effects estimation to obtain fitted values
	        2.1.2 take fitted values (overall and within model) and create
	                  binary variable depending on cutoff value c
	        2.1.3 Compare forecast (binary variable) with realized values and create
	              Graph ROC curves of model performance
		    ## note: important to have different samples and time horizons T in mind
		
		(### no need to replicate ###)
		2.2 Robustness checks
	        2.2.1 Conflict incidence
		    2.2.2 Varying number of topics
		    2.2.3 Topics with other confounders
			  2.2.3.1 Political regime dummies, infant mortality, share of
	                  population descriminated against, dummy if neighboring
	                  countries in conflict
			  2.2.3.2 Conflict escalation: ongoing armed conflict as
	                  predictor for civil war
		    2.2.4 Changing definition of conflict
			  2.2.4.1 all types of conflict (+external wars)
			  2.2.4.2 battle-related deaths in internal wars
			  2.2.4.3 data by Besley & Persson (2011b)
			  2.2.4.4 conflict if >0.08 battle death per 1000 inhabitants
			  2.2.4.5 upper bound estimations PRIO
			  2.2.4.6 UNHCR refugees
			  2.2.4.7 Two years forecast horizon
		    2.2.5 By space and time
		    2.2.6 Comparison to conflict events and keyword counts
		    2.2.7 Neural-network approach
		    2.2.8 Logit without country fixed effects

	(### maybe relevant for later but not now ###)
    3) Analysis why topics provide useful forecasting power on time dimension 
		3.1 Topics harmonized to baseline year 2013
		3.2 LASSO (three parameters of selectivity: 100, 150 & 200 for both y
		3.3 Robustness checks (excluding conflict topics)
'''

################################################################################
######################>>>>>>>END NOTES SECTION<<<<<<<<<#########################
################################################################################


################################################################################
#########################>>>>>>>>MODEL NOTES<<<<<<<<<###########################
################################################################################

# Conflict Types
#	Stata Variable Name: "conflict_type"
#		2 = Armed Conflict = 'bdbest25'
#		3 = Civil Way = 'bdbest1000'

# Incidence vs. Onset
#	Stata Variable Name: "cheat" / "included"
#		1 = onset
#		NULL = incidence
#		Remove 1 or NULL for dependent variable

################################################################################
######################>>>>>>>>>END MODEL NOTES<<<<<<<<<#########################
################################################################################