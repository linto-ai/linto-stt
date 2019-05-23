> Server stt-standalone-worker Endpoints
>  -
* '/transcribe': 
	* methods='POST'
	* inputs='wavFile'
	* output=audio transcription
* '/check':
	* methods='GET'
	* inputs=None
	* output=1 if the service is on
* '/stop':
	* methods='POST'
	* inputs=None
	* output=None

> Config:
>  -
* Environment variables:
	* SERVICE_PORT
* Volumes:
	* /path/to/the/AM/directory:/opt/models/AM
	* /path/to/the/LM/directory:/opt/models/LM


> Parameters
> -
AM config file (decode.cfg)
* decoder: Acoustic model used for decoding (dnn2, dnn3)
* ampath: Path to the AM directory
* lmPath: Path to the language model generation directory (this option is only required by the model service)
* lmOrder: Ngrams used to generate the language model (this option is only required by the model service)
* min_active: Min active tokens
* max_active: Max active tokens
* beam: Decoding beam
* lattice_beam: Beam used in lattice generation
* acwt: Acoustic likelihoods scale
* frame_subsampling_factor : related to chain models (dnn3)

AM directory:
*	final.mdl
*	conf
	*	ivector_extractor.conf
	*	mfcc.conf
	*	online_cmvn.conf
	*	splice.conf
	*	online.conf
*	ivector_extractor
	*	final.dubm
	*	final.ie
	*	final.mat
	*	global_cmvn.stats
	*	online_cmvn.conf
	*	splice_opts

> Nb:  use relative path in config files
> example: decode.cfg
	>ampath: online/
	>lmPath: online/LM_gen/

> Nb: ivector_extractor.conf must include only the extraction parameters without the file dependecies
	> - --num-gselect=5
	>- --min-post=0.025
	>- --posterior-scale=0.1
	>- --max-remembered-frames=1000
	>- --max-count=100
	
LM directory:
*	HCLG.fst
*	words.txt

