jQuery(document).ready(function($) {
	$("#loadButton").click(function(){
	  // Read selectors
	  var selectedDataset = $("#datasetSelection").val();
	  var selectedGroup = $("#groupSelection").val();
	  var selectedColumn = $("#columnSelection").val();
	  var selectedTruth = $("#truthSelection").val();
	  var selectedSample = $("#sampleSelection").val();
	  var selectedModel = $("#modelSelection").val();
	
	  //console.log(selectedDataset);
	
				
	  var selections = {
				"dataset": selectedDataset, 
			    "group": selectedGroup,
				"column": selectedColumn,
				"truth": selectedTruth,
				"sample": selectedSample,
				"model": selectedModel
				};
	
				
	  $.ajax({
		url: '/viewer',
		data: JSON.stringify(selections),
		contentType: 'application/json',
		dataType: 'json',
		type: 'POST',
		success: function (response) {

			console.log("response: " + JSON.stringify(response));

			if (response.subvolumes.length==0) {
				console.log("no images meet the criteria")

				var block_copy = $('#shadowViewerEntry').clone();
				
				block_copy.attr("style", "display: none !important;")
				
				// Empty the gallery
	  			$("#viewer-collection").empty();
	
				// then append
				($('#viewer-collection')).append(block_copy);
			} else {
	
				var viewer_blocks = [];
	
				$.each( response.subvolumes, function(index, subvolume ) {
					// Make a clone of a shadow block 
					var block_copy = $('#shadowViewerEntry').clone();
	
					var plotlymono_img = JSON.stringify(subvolume.plotlymonoImg);
					var plotlycolor_img = JSON.stringify(subvolume.plotlycolorImg);
					var ytcolor_img = JSON.stringify(subvolume.ytcolorImg);
					var gradcam_img = JSON.stringify(subvolume.gradcamImg);
					var gradcamrev_img = JSON.stringify(subvolume.gradcamrevImg);
	
	
					block_copy.attr("style", "display: block !important;");
					block_copy.find('#plotlymono-img').attr("src", plotlymono_img.replace(/\"/g, ""));
					block_copy.find('#plotlycolor-img').attr("src", plotlycolor_img.replace(/\"/g, ""));
					block_copy.find('#ytcolor-img').attr("src", ytcolor_img.replace(/\"/g, ""));
					block_copy.find('#gradcam-img').attr("src", gradcam_img.replace(/\"/g, ""));
					block_copy.find('#gradcamrev-img').attr("src", gradcamrev_img.replace(/\"/g, ""));

					
					viewer_blocks.push(block_copy);
				});

				console.log("viewer blocks include: " + viewer_blocks)
	
				// Empty the gallery
	  			$("#viewer-collection").empty();
	
				// then append
				($('#viewer-collection')).append(viewer_blocks);		

				
			}
		},
		error: function(error) {
			console.log(error);
		}
	  });
	});
});


