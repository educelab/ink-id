jQuery(document).ready(function($) {
	$("#loadButton").click(function(){
	  // Read selectors
	  var selectedDataset = $("#datasetSelection").val();
	  var selectedGroup = $("#groupSelection").val();
	  var selectedColumn = $("#columnSelection").val();
	  var selectedTruth = $("#truthSelection").val();
	  var selectedImagetype = $("#imageSelection").val()
	  var selectedPrediction = $("#predictionSelection").val();
	  var selectedModel = $("#modelSelection").val();
	
	  //console.log(selectedDataset);
	
	  // Delete the previous selection
	  $("#selection1 #selection2 #selection3 #selection4 #selection5 #selection6 #selection7").empty();
	
	  // Display the current selection
	  $("#selection1").text("Dataset: " + selectedDataset);
	  $("#selection2").text("Group: " + selectedGroup);
	  $("#selection3").text("Column: " + selectedColumn);
	  $("#selection4").text("Truth: " + selectedTruth);
	  $("#selection5").text("Image Type:" + selectedImagetype);
	  $("#selection6").text("Prediction: " + selectedPrediction);
	  $("#selection7").text("Model: " + selectedModel);
	
	
	
	  // This will be replaced by the REST API list
	  /*
	  var inkImages = ['/static/images/gradcam1.png', '/static/images/gradcam2.png', 
	  	   '/static/images/gradcam3.png', '/static/images/plotly_test.png',
	                   '/static/images/gradcam2.png'];
	  var dataFiles = ['/static/texts/metadata1.json', 
			   '/static/texts/metadata1.json',
			   '/static/texts/metadata1.json',
			   '/static/texts/metadata1.json',
			   '/static/texts/metadata1.json'];
	  */
				
	  var selections = {
				"dataset": selectedDataset, 
			    "group": selectedGroup,
				"column": selectedColumn,
				"truth": selectedTruth,
			    "prediction": selectedPrediction,
			    "imagetype": selectedImagetype,
				"model": selectedModel};
	
	  /*
	  var selections = '{"dataset": "sampleDataset", "truth": "NoInk"}';
	  */ 
				
	  $.ajax({
		url: '/filter',
		data: JSON.stringify(selections),
		contentType: 'application/json',
		dataType: 'json',
		type: 'POST',
		//headers: { 'Content-Type': 'application/json' },
		success: function (response) {
		
			console.log("data type of response is: " + typeof response);
	
			if (response.paths.length==0) {
				console.log("no images meet the criteria")
	
				// Even then, gallery has to be cleared
				var block_copy = $('#shadowBlock').clone();
	
				block_copy.attr("style", "display: none !important;")
				// Empty the gallery
	  			$("#gallery").empty();
	
				// then append
				($('#gallery')).append(block_copy);
	
			} else {
				//console.log("Received response.paths is: " + JSON.stringify(response.paths));
				//console.log("Received rsponse.paths[0] is: " + JSON.stringify(response.paths[0]));
				//console.log("Received response.path[0].image-path is: " +  JSON.stringify(response.paths[0].image));
				
				// Make a clone of a shadow block 
				//var block_copy = $('#shadowBlock').clone();
	
				var gallery_blocks = [];
	
	
				$.each( response.paths, function(index, path ) {
					// Make a clone of a shadow block 
					var block_copy = $('#shadowBlock').clone();
	
					var imagePath = JSON.stringify(path.image);
					var metadataPath = JSON.stringify(path.metadata);
	
					console.log("path.image is: " + imagePath);
	
					block_copy.attr("style", "display: inline !important;");
					block_copy.find('#popup-image').attr("href", imagePath.replace(/\"/g, ""));
					block_copy.find('#thumbnail-image').attr("src", imagePath.replace(/\"/g, ""));
					block_copy.find('#metadata-json').attr("href", metadataPath.replace(/\"/g, ""));
					block_copy.find('#metadata-path').text("metadata json");
					
					gallery_blocks.push(block_copy);
				});
				/*	
				test1.attr("style", "display: inline !important;");
				test1.find('#popup-image').attr("href", inkImages[0]);
				test1.find('#thumbnail-image').attr("src", inkImages[0]);
				test1.find('#metadata-json').attr("href", "/static/texts/metadata1.json");
				test1.find('#metadata-path').text("/static/texts/metadata1");
	  			*/
	
				// Empty the gallery
	  			$("#gallery").empty();
	
				// then append
				($('#gallery')).append(gallery_blocks);		
			}
		},
		error: function(error) {
			console.log(error);
		}
	  });
	});
});


