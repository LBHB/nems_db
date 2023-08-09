$(document).ready(function(){

    // TODO: Split this up into multile .js files? getting a bit crowded in here,
    // could group by functionality at this point.

    // Socketio functionality disabled for now.
    /*
    namespace = '/py_console'

    var socket = io.connect(
            location.protocol + '//'
            + document.domain + ':'
            + location.port + namespace,
            {'timeout':0}
            );
    var socket = io.connect(
            null,
            {   port: location.port,
                rememberTransport: false
            }
            )

    socket.on('connect', function() {
       console.log('socket connected');
    });

    socket.on('console_update', function(msg){
        var color = '';
        // TODO: add special color markers here for other messages?
        if ((msg.data === 'Login required') || (msg.data === 'Fit failed.')){
            color = 'style="color: red"';
        }
        $('#py_console').prepend(
                "<p class='py_con_msg'" + color + ">" + msg.data + "</p>"
                );
    });

    */
    // use this in place of console.log to send to py_console
    function py_console_log(message){
        var color = '';
        // TODO: add special color markers here for other messages?
        if ((message === 'Login required') || (message === 'Fit failed.')){
            color = 'style="color: red"';
        }
        $('#py_console').prepend(
                "<p class='py_con_msg'" + color + ">" + message + "</p>"
                );
    }

    //initializes bootstrap popover elements
    $('[data-toggle="popover"]').popover({
        trigger: 'click',
    });

    $("#displayRow").resizable({
        handles: "w, e"
    });

    $("#tableRow").resizable({
        handles: "w, e"
    })

    //$("#selectionsRow").resizable({
    //    handles: "n, w, e, s"
    //});

    //$("#pyConRow").resizable({
    //    handles: "n, s"
    //});

    /*
    //drags start out disabld until alt is pressed
    $(".dragToggle").draggable('disable');

    $("#enableDraggable").on('click', function(){
        $(".dragToggle").draggable('enable');
        return false;
    });
    $("#disableDraggable").on('click', function(){
        $(".dragToggle").draggable('disable');
        return false;
    });
    */

    /*
    function initTable(table){
        // Called any time results is updated -- set table options here
        table.DataTable({
            paging: false,
            search: {
                "caseInsensitive": true,
            },
            //responsive: true,
            // TODO: add select option to replace custom one
            // (can add ctrl/shift click etc)
        });
    }
    */

    var cols_array = []
    function get_default_cols(){
        $("#defaultColsDiv").children().each(function(){
            cols_array.push($(this).val());
        });
        $("#tableColSelector").val(cols_array);
    }
    get_default_cols();

    class Saved_Selections {
        constructor(){
            this.tags = '__any';
            this.status = '__any';
            this.analysis = 'nems testing';
            this.plot_measure = 'r_test';
            this.plot_type = 'Scatter';
            this.script = 'demo_script';
            this.onlyFair = 1;
            this.includeOutliers = 0;
            this.snri = $("#default_snri").val();
            this.snr = $("#default_snr").val();
            this.iso = $("#default_iso").val();
            this.cols = cols_array;
            this.sort = 'cellid';
            this.row_limit = 500;
            this.code_hash = '';
            this.exec_path = '';
            this.script_path = '';
            this.extra_models = '';
            this.extra_analyses = '';
            this.kamiakFunction = '';
            this.kamiakPath = '';
            this.kamiakResults = '';
        }
    }

    var saved_selections = new Saved_Selections();
    // call on page load
    get_saved_selections();

    function get_saved_selections(){
        $.ajax({
            url: $SCRIPT_ROOT + '/get_saved_selections',
            data: {},
            type: 'GET',
            success: function(data){
                // data.null will be true if user is not logged in
                // or if other issue happens in flask route function
                if (data.null === false){
                    console.log("retrieved selections");
                    saved = JSON.parse(data.selections);
                    keys = Object.keys(saved);
                    //console.log("retrieved selections: " + keys);
                    for (i=0; i<keys.length; i++){
                        //console.log("key: " + keys[i] + ", value: " + saved[keys[i]])
                        saved_selections[keys[i]] = saved[keys[i]];
                    }
                } else {
                    console.log("no selections to load");
                }
                assign_selections();
            },
            error: function(error){
                console.log(error);
                assign_selections();
            }
        });
    }

    function set_saved_selections(){
        $.ajax({
            url: $SCRIPT_ROOT + '/set_saved_selections',
            data: {stringed_selections:JSON.stringify(saved_selections)},
            type: 'GET',
            success: function(data){
                if (data.null === false){
                    console.log('user selections saved successfully');
                } else {
                    console.log("Couldn't save selections -- make sure you are logged in.")
                }
            },
            error: function(error){
                console.log('error when saving user selections');
            }
        });
    }

    function assign_selections(){
        $("#tagFilters").val(saved_selections.tags);
        $("#statusFilters").val(saved_selections.status).change();

        $("#codeHash").val(saved_selections.code_hash);

        $("#rowLimit").val(saved_selections.row_limit);
        $("#tableSortSelector").val(saved_selections.sort);
        $("#tableColSelector").val(saved_selections.cols).change();

        if (saved_selections.onlyFair === 1){
            document.getElementById('onlyFair').checked = true;
        } else{
            document.getElementById('onlyFair').checked = false;
        }

        if (saved_selections.includeOutliers === 1){
            document.getElementById('includeOutliers').checked = true;
        } else{
            document.getElementById('includeOutliers').checked = false;
        }

        $("#plotTypeSelector").val(saved_selections.plot_type);
        $("#measureSelector").val(saved_selections.plot_measure);
        $("#customSelector").val(saved_selections.script);

        snr = saved_selections.snr;
        snri = saved_selections.snri;
        iso = saved_selections.iso;

        updatePlotOpVal();

        // temporary solution but not great since time to finish ajax calls might
        // be longer in some cases.
        //setTimeout(function(){ wait_on_analysis = false; }, 3000);
        $("#analysisSelector").val(saved_selections.analysis);
        $("[name='extraModels']").val(saved_selections.extra_models);
        $("[name='extraAnalyses']").val(saved_selections.extra_analyses).change();

        $("#scriptPath").val(saved_selections.script_path).change();
        $("#execPath").val(saved_selections.exec_path).change();
        $("#kamiakFunction").val(saved_selections.kamiakFunction).change();
        $("#kamiakPath").val(saved_selections.kamiakPath).change();
        $("#kamiakResults").val(saved_selections.kamiakResults).change();

        $("#exaOHSU").val(saved_selections.exaOHSU).change();
        $("#exaExec").val(saved_selections.exaExec).change();
        $("#exaScript").val(saved_selections.exaScript).change();
        $("#exaLimit").val(saved_selections.exaLimit).change();
        $("#exaExclude").val(saved_selections.exaExclude).change();
    }



    // saved_selections updater functions
    $("#analysisSelector").change(function(){
        saved_selections.analysis = $(this).val();
    });
    $("#rowLimit").change(function(){
        saved_selections.row_limit = $(this).val();
    });
    $("#tagFilters").change(function(){
        saved_selections.tags = $(this).val();
    });
    $("#statusFilters").change(function(){
        saved_selections.status = $(this).val();
    });
    $("#tableColSelector").change(function(){
        saved_selections.cols = $(this).val();
    });
    $("#tableSortSelector").change(function(){
        saved_selections.sort = $(this).val();
    });
    $("#onlyFair").change(function(){
        if (document.getElementById("onlyFair").checked){
            saved_selections.onlyFair = 1;
        } else {
            saved_selections.onlyFair = 0;
        }
    });
    $("#includeOutliers").change(function(){
        if (document.getElementById("includeOutliers").checked){
            saved_selections.includeOutliers = 1;
        } else {
            saved_selections.includeOutliers = 0;
        }
    });
    $("#plotTypeSelector").change(function(){
        saved_selections.plot_type = $(this).val();
    });
    $("#measureSelector").change(function(){
        saved_selections.plot_measure = $(this).val();
    });

    $("#scriptPath").change(function(){
        saved_selections.script_path = $(this).val();
    });

    $("#execPath").change(function(){
        saved_selections.exec_path = $(this).val();
    });

    $("#kamiakFunction").change(function(){
        saved_selections.kamiakFunction = $(this).val();
    });

    $("#kamiakPath").change(function(){
        saved_selections.kamiakPath = $(this).val();
    });

    $("#kamiakResults").change(function(){
        saved_selections.kamiakResults = $(this).val();
    });

    $("#exaOHSU").change(function(){
        saved_selections.exaOHSU = $(this).val();
    });

    $("#exaExec").change(function(){
        saved_selections.exaExec = $(this).val();
    });

    $("#exaScript").change(function(){
        saved_selections.exaScript = $(this).val();
    });

    $("#exaLimit").change(function(){
        saved_selections.exaLimit = $(this).val();
    });

    $("#exaExclude").change(function(){
        saved_selections.exaExclude = $(this).val();
    });

    $("[name='extraModels']").change(function(){
        saved_selections.extra_models = $(this).val();
    });

    $("[name='extraAnalyses']").change(function(){
        saved_selections.extra_analyses = $(this).val();
    });

    // snri, snr and iso handled in their own section since they aren't DOM elements

    // save user selections on refresh, window close or navigate away
    window.onbeforeunload = save_before_close;
    function save_before_close(){
        set_saved_selections();
    }
    // also save selections every 60 seconds just to be safe
    // TODO: 60 seconds okay, or would shorter/longer be better?
    //setInterval(function(){
    //    set_saved_selections();
    //    console.log("Selections saved.");
    //    },
    //    60000
    //);

    $("#testSave").click(set_saved_selections);
    $("#testGet").click(get_saved_selections);
    $("#testPrint").click(function(){
        py_console_log(saved_selections.analysis + " " + saved_selections.tags);
    });


    // is this needed anymore? loading saved selections should preclude this
    /*
    var analysisCheck = document.getElementById("analysisSelector").value;
    if ((analysisCheck !== "") && (analysisCheck !== undefined) && (analysisCheck !== null)){
        updateBatchModel();
        updateAnalysisDetails();
    }
    */


    $("#selectAllCells").on('click', selectCellsCheck);
    function selectCellsCheck(){
        var cellOptions = document.getElementsByName("cellOption[]");
        for (var i=0;i<cellOptions.length;i++){
            if (document.getElementById("selectAllCells").checked){
                cellOptions[i].selected = true;
            } else{
                cellOptions[i].selected = false;
            }
        }
        updateResults();
    }

    $("#selectAllModels").on('click', selectModelsCheck);
    function selectModelsCheck(){
        var modelOptions = document.getElementsByName("modelOption[]");
        for (var i=0;i<modelOptions.length;i++){
            if (document.getElementById("selectAllModels").checked){
                modelOptions[i].selected = true;
            } else{
                modelOptions[i].selected = false;
            }
        }
        updateResults();
    }


    $("#analysisSelector").change(updateBatchModel);
    $("[name='extraModels']").change(updateBatchModel);
    $("[name='extraAnalyses']").change(updateBatchModel);
    $("#modelSearch").change(updateBatchModel);

    function updateBatchModel(){
        // if analysis selection changes, get the value selected
        var aSelected = $("#analysisSelector").val();
        var extraModels = $("[name='extraModels']").val();
        var extraAnalyses = $("[name='extraAnalyses']").val();
        var modelSearch = $("#modelSearch").val();

        // pass the value to '/update_batch' in nemsweb.py
        // get back associated batchnum and change batch selector to match
        $.ajax({
            url: $SCRIPT_ROOT + '/update_batch',
            data: { aSelected:aSelected },
            type: 'GET',
            success: function(data) {
                if ((data.blank === 1) || (data.blank === "1")){
                    $("#batchSelector").val($("#batchSelector option:first").val()).change();
                } else {
                    $("#batchSelector").val(data.batch).change();
                }
            },
            error: function(error) {
                console.log(error);
            }
        });

        // also pass analysis value to 'update_models' in nemsweb.py
        $.ajax({
            url: $SCRIPT_ROOT + '/update_models',
            data: { aSelected:aSelected, extraModels:extraModels,
                    extraAnalyses:extraAnalyses, modelSearch:modelSearch},
            type: 'GET',
            success: function(data){
                if (data.modellist.length === 0){
                    console.log('No model list returned.');
                    py_console_log('No model list returned.');
                    return false;
                }
                var models = $("#modelSelector");
                models.empty();

                $.each(data.modellist, function(i, modelname){
                    models.append($("<option></option>")
                        .attr("value", modelname)
                        .attr("name","modelOption[]")
                        .text(modelname));
                });

                selectModelsCheck();
            },
            error: function(error){
                console.log(error);
            }
        });
    };

    $("#batchSelector").change(updateCells);
    $("#cellSearch").change(updateCells);

    function updateCells(){
        // TODO: update cell list when batch changes
        //       should cascade from change to analysis selection
        // if batch selection changes, get the value of the new selection
        var bSelected = $("#batchSelector").val();
        var aSelected = $("#analysisSelector").val();
        var cellSearch = $("#cellSearch").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_cells',
            data: { bSelected:bSelected, aSelected:aSelected,
                    cellSearch:cellSearch },
            type: 'GET',
            success: function(data) {
                cells = $("#cellSelector");
                cells.empty();

                $.each(data.celllist, function(cell) {
                    cells.append($("<option></option>")
                        .attr("value", data.celllist[cell])
                        .attr("name","cellOption[]")
                        .text(data.celllist[cell]));
                });

                selectCellsCheck();
            },
            error: function(error) {
                console.log(error);
            }
        });
    };

    function addLinksToTable(){
        // Iterate through each table row and convert each result
        // to a link to detailed info for that result
        var $table = $("#tableWrapper").children('table');
        $table.find('tbody').find('tr').each(function(){
            var cellid = $(this).children().eq(0).html();
            var modelname = $(this).children().eq(1).html();
            var cell_link = $SCRIPT_ROOT + '/cell_details/';
            var model_link = $SCRIPT_ROOT + '/model_details/';
            $(this).children().eq(0).html(
                    "<p cellid='" + cellid + "'>" + cellid + "</p>"
                    );
                    // can switch this back in when figure out a good way to toggle
                    // links on and off
                    //"<a href='" + cell_link + cellid + "' target='_blank'"
                    //+ "id='" + cellid + "'>" + cellid + "</a>"
            $(this).children().eq(1).html(
                    "<p modelname='" + modelname + "'>" + modelname + "</p>"
                    );
                    //"<a href='" + model_link + modelname + "' "
                    //+ "id='" + modelname + "'>" + modelname + "</a>"
        });
    }

    $("#modelSelector,#cellSelector,#rowLimit,#tableColSelector,#tableSortSelector,#descending")
    .change(updateResults);
    function updateResults(){
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var colSelected = $("#tableColSelector").val();
        var sortSelected = $("#tableSortSelector").val();
        if (document.getElementById("descending").checked){
            var ordSelected = "desc";
        } else {
            var ordSelected = "asc";
        }
        var rowLimit = $("#rowLimit").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_results',
            data: { bSelected:bSelected, cSelected:cSelected,
                   mSelected:mSelected, colSelected:colSelected,
                   rowLimit:rowLimit, ordSelected:ordSelected,
                   sortSelected:sortSelected },
            type: 'GET',
            success: function(data) {
                //grabs whole div - replace inner html with new table?
                results = $("#tableWrapper");
                //results.resizable("destroy");
                //results.draggable("destroy");
                results.html(data.resultstable)
                //sizeDragTable();
                var table = results.children("table");
                //initTable(table);
                //disabled for now - need to figure out agood way to let user
                // toggle the links on and off
                addLinksToTable();
            },
            error: function(error) {
                console.log(error);
            }
        });
    }

    updateColText();
    $("#tableColSelector").change(updateColText);
    function updateColText(){
        button = $("#colsModalButton");
        text = $("#tableColSelector").val();
        button.html("");
        for (i=0; i < text.length; i++){
            button.append(text[i] + ', ');
            if (i >= 4){
                button.append('...');
                break;
            }
        }
        if (text.length === 0){
            button.html("Columns");
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //         Analysis details, tags/status, edit/delete/new             //
    ////////////////////////////////////////////////////////////////////////


    $("#analysisSelector").change(updateAnalysisDetails);

    function updateAnalysisDetails(){
        var aSelected = $("#analysisSelector").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_analysis_details',
            data: { aSelected:aSelected },
            type: 'GET',
            success: function(data) {
                $("#analysisDetails").attr("data-content",data.details)
                $("#analysisDetails").attr("title",aSelected)
            },
            error: function(error) {
                console.log(error);
            }
        });
    }

    updateAnalysis();
    $("#tagFilters, #statusFilters").change(updateAnalysis);

    function updateAnalysis(){
        analysis_still_updating = true;
        var tagSelected = $("#tagFilters").val();
        var statSelected = $("#statusFilters").val();
        $.ajax({
           url: $SCRIPT_ROOT + '/update_analysis',
           data: { tagSelected:tagSelected, statSelected:statSelected },
           type: 'GET',
           success: function(data){
                analyses = $("#analysisSelector");
                analyses.empty();

                $.each(data.analysislist, function(analysis) {
                    analyses.append($("<option></option>")
                        .attr("value", data.analysislist[analysis])
                        .attr("name", data.analysis_ids[analysis])
                        .text(data.analysislist[analysis]));
                });

                if (data.analysislist.includes(saved_selections.analysis)){
                    $("#analysisSelector").val(saved_selections.analysis).change();
                } else {
                    $("#analysisSelector").val($("#analysisSelector option:first").val()).change();
                }
           },
           error: function(error){
                console.log(error);
           }
        });
    }


    updateStatusText();
    updateTagText();

    $("#statusFilters").change(updateStatusText);
    $("#tagFilters").change(updateTagText);

    function updateStatusText(){
        button = $("#statusModalButton");
        text = $("#statusFilters").val();
        button.html("");
        for (i=0; i<text.length; i++){
            if (!(text[i] === '__any')) {
                button.append(text[i] + ", ");
            } else {
                button.html('Status');
                break;
            }
            if (i >= 4){
                button.append('...');
                break;
            }
        }
    }

    function updateTagText(){
        button = $("#tagModalButton");
        text = $("#tagFilters").val();
        button.html("");
        for (i=0; i<text.length; i++){
            if (!(text[i] === '__any')) {
                button.append(text[i] + ", ");
            } else {
                button.html('Tags');
                break;
            }
            if (i >= 4){
                button.append('...');
                break;
            }
        }
    }

    /*
    function updateTagOptions(){
        $.ajax({
           url: $SCRIPT_ROOT + '/update_tag_options',
           type: 'GET',
           success: function(data){
               tags = $("#tagFilters")
               tags.empty();
               tags.append(
                        "<ul class='list-unstyled col-sm-6'>"
                        + "<li>"
                        + "<input class='statusOption' name='statusOption[]' type="
                        + "'radio' value='__any' checked>"
                        + "<span class='lbl'> Any </span>"
                        + "</li>"
                        )

               $.each(data.taglist, function(tag){
                   if (tag%12 == 0){
                       tags.append(
                                "</ul><ul class='list-unstyled col-sm-6'>"
                                )
                   }

                   tags.append(
                           "<li>"
                            + "<input class='tagOption'"
                            + "name='tagOption[]' type='radio' value="
                            + data.taglist[tag] + "'" + ">"
                            + "<span class='lbl'>" + data.taglist[tag] + "</span>"
                            + "</li>"
                            )
               });
               tags.append("</ul>");

           },
           error: function(error){
               console.log(error);
           }
        });
    }

    function updateStatusOptions(){
        $.ajax({
           url: $SCRIPT_ROOT + '/update_status_options',
           type: 'GET',
           success: function(data){
               statuses = $("#statusFilters")
               statuses.empty();
               statuses.append(
                        "<li>"
                        + "<input class='statusOption' name='statusOption[]' type="
                        + "'radio' value='__any' checked>"
                        + "<span class='lbl'> Any </span>"
                        + "</li>"
                        )

               $.each(data.statuslist, function(status){
                   if (status == 0){
                       var checked = 'checked'
                   } else{
                       var checked = ''
                   }
                   statuses.append(
                            "<li>"
                            + "<input class='tagOption'"
                            + "name='tagOption[]' type='radio' value="
                            + data.statuslist[status] + "'" + ">"
                            + "<span class='lbl'>" + data.statuslist[status]
                            + "</span>"
                            + "</li>"
                            )
               });
           },
           error: function(error){
               console.log(error);
           }
        });
    }
    */

    $("#newAnalysis").on('click',newAnalysis);

    function newAnalysis(){
        $("[name='editName']").val('');
        $("[name='editId']").val('__none');
        $("[name='editStatus']").val('');
        $("[name='editTags']").val('');
        $("[name='editQuestion']").html('');
        $("[name='editAnswer']").html('');
        $("[name='editLoadTree']").val('');
        $("[name='editModTree']").val('');
        $("[name='editFitTree']").val('');
        $("[name='editExtraModels']").val('');
    }

    $("#editAnalysis").on('click',editAnalysis);

    function editAnalysis(){
        // get current analysis selection
        // ajax call to get back name, tags, question, etc
        // fill in content of editor modal with response

        var aSelected = $("#analysisSelector option:selected").attr('name');

        $.ajax({
            url: $SCRIPT_ROOT + '/get_current_analysis',
            data: { aSelected:aSelected },
            type: 'GET',
            success: function(data){
                $("[name='editName']").val(data.name);
                $("[name='editId']").val(data.id);
                $("[name='editStatus']").val(data.status);
                $("[name='editTags']").val(data.tags);
                $("[name='editQuestion']").html(data.question);
                $("[name='editAnswer']").html(data.answer);
                $("[name='editLoadTree']").val(data.load);
                $("[name='editModTree']").val(data.mod);
                $("[name='editFitTree']").val(data.fit);
                $("[name='editExtraModels']").val(data.model_extras);
            },
            error: function(error){
                console.log(error);
            }
        });
    }

    $("#submitEditForm").on('click',verifySubmit);


    // TODO: change this to run nested ajax call instead of .submit()
    // so that entire page doesn't have to refresh afterward. should only have
    // to call updateAnalysis() to refresh the list for analysis selector.

    function verifySubmit(){
        var nameEntered = $("[name='editName']").val();
        var analysisId = $("[name='editId']").val();

        $.ajax({
           url: $SCRIPT_ROOT + '/check_analysis_exists',
           data: { nameEntered:nameEntered, analysisId:analysisId },
           type: 'GET',
           success: function(data){
                if (data.exists){
                    alert("An analysis by the same name already exists.\n" +
                          "Please choose a different name.");
                    return false;
                }

                if(confirm("ATTENTION: This will save the entered information to the\n" +
                            "database, potentially overwriting previous settings.\n" +
                            "Are you sure you want to continue?")){
                    submitAnalysis();

                } else{
                    return false;
                }
            },
           error: function(error){

            }
        });
    }

    function submitAnalysis(){
        var name = $("[name='editName']").val();
        var id = $("[name='editId']").val();
        var status = $("[name='editStatus']").val();
        var tags = $("[name='editTags']").val();
        var question = $("[name='editQuestion']").val();
        var answer = $("[name='editAnswer']").val();
        var load = $("[name='editLoadTree']").val();
        var mod = $("[name='editModTree']").val();
        var fit = $("[name='editFitTree']").val();
        var model_extras = $("[name='editExtraModels']").val();

        $.ajax({
           url: $SCRIPT_ROOT + '/edit_analysis',
           data: { name:name, id:id, status:status, tags:tags,
                  question:question, answer:answer, load:load,
                  mod:mod, fit:fit, model_extras:model_extras },
           type: 'GET',
           success: function(data){
               $("#analysisEditorModal").modal('hide')
               py_console_log(data.success);
               saved_selections.analysis = name;
               updateAnalysis();
           },
           error: function(error){
               console.log(error)
           }
        });
    }

    $("#deleteAnalysis").on('click',deleteAnalysis);

    function deleteAnalysis(){

        var aSelected = $("#analysisSelector option:selected").attr('name');
        var aName = $("#analysisSelector").val();
        reply = confirm("WARNING: This will delete the database entry for the selected " +
                "analysis. \n\n!!   THIS CANNOT BE UNDONE   !!\n\n" +
                "Are you sure you wish to delete this analysis:\n" +
                aName);

        if (reply){
            $.ajax({
                url: $SCRIPT_ROOT + '/delete_analysis',
                data: { aSelected:aSelected },

                // TODO: should use POST here? but was causing issues

                type: 'GET',
                success: function(data){
                    if (data.success){
                        py_console_log(aSelected + " successfully deleted.");
                        updateAnalysis();
                    } else{
                        py_console_log("Something went wrong - unable to delete:\n" + aSelected);
                        return false;
                    }
                },
                error: function(error){
                    console.log(error);
                }
            });
        } else{
            return false;
        }
    }



    ///////////////////////////////////////////////////////////////////////
    //     table selection, preview, strf, inspect                       //
    ///////////////////////////////////////////////////////////////////////


    CTRL = false;
    // TODO: implement shift select
    SHIFT = false;
    UPPER = -1;

    $(document).keydown(function(event){
        if (event.ctrlKey){
            CTRL = true;
        }
        if (event.shiftKey){
            SHIFT = true;
        }
    });
    $(document).keyup(function(event){
        CTRL = false;
        SHIFT = false;
    });

    $(document).on('click','.dataframe tbody tr',function(){
        if ($(this).hasClass('selectedRow')){
            $(this).removeClass('selectedRow');
        } else{
            //comment this out for multi-select
            if (!CTRL){
                $(this).parents('tbody').find('.selectedRow').each(function(){
                    $(this).removeClass('selectedRow');
                });
            }
            $(this).addClass('selectedRow');
            if (document.getElementById("autoPreview").checked){
                refreshPreview();
            }
        }
    });

    $(document).on('click','#preview', refreshPreview);
    function refreshPreview(){
        var cSelected = [];
        var mSelected = [];
        var bSelected = $("#batchSelector").val();

        // have to change .children('p')  back to 'a' if table links put back in
        $(".dataframe tr.selectedRow").each(function(){
            cSelected.push($(this).children().eq(0).children('p').attr('cellid'));
        });
        $(".dataframe tr.selectedRow").each(function(){
            mSelected.push($(this).children().eq(1).children('p').attr('modelname'));
        });

        // only proceed if selections have been made
        if ((cSelected.length == 0) || (mSelected.length == 0)){
            py_console_log('Must select at least one result from table')
            return false;
        }

        $.ajax({
            url: $SCRIPT_ROOT + '/get_preview',
            data: { cSelected:cSelected, mSelected:mSelected,
                   bSelected:bSelected },
            type: 'GET',
            success: function(data){
                //$("#displayWrapper").resizable("destroy");
                //$("displayWrapper").draggable("destroy");
                $("#displayWrapper").html(
                        '<img id="preview_image" src="data:image/png;base64,'
                        + data.image + '" />'
                        );
                //sizeDragDisplay();
            },
            error: function(error){
                console.log(error);
            }
        });
    };

    $("#clearSelected").on('click',function(){
        $(".dataframe tr.selectedRow").each(function(){
            $(this).removeClass('selectedRow');
        });
    });

    $("#strf").on('click',function(){
        py_console_log("STRF Function not yet implemented");
        //return strf plots ala _analysis
        //low priority
    });




    //////////////////////////////////////////////////////////////////////
    //               Model fitting, inspect                             //
    //////////////////////////////////////////////////////////////////////



    function addLoad(){
        $('#loadingPopup').css('display', 'block');
    }
    function removeLoad(){
        $('#loadingPopup').css('display', 'none');
    }

    var queueCount = 0;
    function addQueue(){
        if (queueCount == 0){
            document.getElementById("enqueue").classList.add('btn-danger');
            document.getElementById("enqueue").classList.remove('btn-default');
        }
        queueCount ++;
    }
    function removeQueue(){
        queueCount --;
        if (queueCount == 0){
            document.getElementById("enqueue").classList.add('btn-default');
            document.getElementById("enqueue").classList.remove('btn-danger');
        }
    }

    $("#toggleFitOp").on('click',function(){
        var fod = $('#fitOpDiv');

        if (fod.css('display') === 'block'){
            fod.css('display', 'none');

        } else if (fod.css('display') === 'none'){
            fod.css('display', 'block');

        } else{
            return false;
        }
    })


/*
    $("#fitSingle").on('click',function(){
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();

        if ((bSelected === null) || (bSelected === undefined) ||
                (bSelected.length == 0)){
            py_console_log('Must select a batch')
            return false;
        }
        if ((cSelected.length > 1) || (mSelected.length > 1) || (cSelected.length
            == 0) || (mSelected.length == 0)){
            py_console_log('Must select one model and one cell')
            return false;
        }

        if (!(confirm(
                "Preparing to fit selection -- this may take several minutes."
              + "Web interface will be disabled until fit is complete."
              + "\n\nSelect OK to continue."))){
            return false;
        }
        // TODO: insert confirmation box here, with warning about waiting for
        //          fit job to finish

        addLoad();

        py_console_log("Sending fit request to server. Success or failure will be"
                    + "reported here when job is finished.")

        $.ajax({
            url: $SCRIPT_ROOT + '/fit_single_model',
            data: { bSelected:bSelected, cSelected:cSelected,
                       mSelected:mSelected },
            // TODO: should use POST maybe in this case?
            type: 'GET',
            timeout: 0,
            success: function(data){
                py_console_log("Fit finished.\n"
                      + "r_test: " + data.r_est + "\n"
                      + "r_val: " + data.r_val + "\n"
                      + "Click 'inspect' to browse model");

                updateResults();
                removeLoad();
                //open preview in new window like the preview button?
                //then would only have to pass file path
                //window.open('preview/' + data.preview,'width=520','height=910')
            },
            error: function(error){
                console.log("Fit failed");
                console.log(error);
                removeLoad();
            },
        });
    });
*/


    $("#enqueue").on('click',function(){
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var codeHash = $("#codeHash").val();
        var execPath = $("#execPath").val();
        var scriptPath = $("#scriptPath").val();
        var kamiakFunction = $("#kamiakFunction").val();
        var kamiakPath = $("#kamiakPath").val();
        var kamiakResults = $("#kamiakResults").val();
        var forceRerun = 0;
        var useKamiak = 0;
        var loadKamiak = 0;
        var useGPU = 0;
        var useExacloud = 0;
        var exaHighMem = 0;
        var exaOHSU = $("#exaOHSU").val();
        var exaExec = $("#exaExec").val();
        var exaScript = $("#exaScript").val();
        var exaLimit = $("#exaLimit").val();
        var exaExclude = $("#exaExclude").val();

        if (document.getElementById('forceRerun').checked){
            forceRerun = 1;
        }

        if (document.getElementById('useKamiak').checked){
            useKamiak = 1;
        }

        if (document.getElementById('loadKamiak').checked){
            loadKamiak = 1;
        }

        if (document.getElementById('useGPU').checked){
            useGPU = 1;
        }

        if (document.getElementById('useExacloud').checked){
            useExacloud = 1;
        }

        if (document.getElementById('exaHighMem').checked){
            exaHighMem = 1;
        }

        if ((bSelected === null) || (bSelected === undefined) ||
                (bSelected.length == 0)){
            py_console_log('Must select a batch')
            return false;
        }
        if ((cSelected.length == 0) || (mSelected.length == 0)){
            py_console_log('Must select at least one model and at least one cell')
            return false;
        }

        var total = cSelected.length * mSelected.length;
        if (!(confirm('This will add ' + total + ' models to the queue.'
                      + '\n\nAre you sure you wish to continue?'))){
            return false;
        }

        if (document.getElementById('enqueue').classList.contains('btn-danger')){
            if (!(confirm('WARNING: You still have other jobs waiting to be processed.'
                          + '\n\nDo you still wish to continue? (Sorry for being annoying!'))){
                return false;
            }
        }

        addQueue();
        py_console_log("Sending fit request for each combination - please wait...");

        $.ajax({
            url: $SCRIPT_ROOT + '/enqueue_models',
            data: { bSelected:bSelected, cSelected:cSelected,
                   mSelected:mSelected, forceRerun, codeHash:codeHash,
                   execPath:execPath, scriptPath:scriptPath,
                   useKamiak:useKamiak, kamiakFunction:kamiakFunction,
                   kamiakPath:kamiakPath, loadKamiak:loadKamiak,
                   kamiakResults:kamiakResults, useGPU:useGPU,
                   useExacloud:useExacloud, exaOHSU:exaOHSU,
                   exaExec:exaExec, exaScript:exaScript, exaLimit:exaLimit,
                   exaExclude:exaExclude, exaHighMem:exaHighMem},
            // TODO: should POST be used in this case?
            type: 'GET',
            success: function(result){
                py_console_log(result);
                removeQueue();
            },
            error: function(error){
                console.log(error)
                removeQueue();
            }
        });
        //communicates with daemon to queue model fitting for each selection on cluster,
        //using similar process as above but for multiple models and no
        //dialogue displayed afterward

        //open separate window/tab for additional specifications like priority?
    });

    $("#inspect").on('click',function(){
        //pull from results table
        var cSelected = [];
        var mSelected = [];
        var bSelected = $("#batchSelector").val();

        $(".dataframe tr.selectedRow").each(function(){
            cSelected.push($(this).children().eq(0).children('a').attr('id'));
            if (cSelected.length > 0){
                return false;
            }
        });
        $(".dataframe tr.selectedRow").each(function(){
            mSelected.push($(this).children().eq(1).children('a').attr('id'));
            if (mSelected.length > 0){
                return false;
            }
        });

        if ((cSelected.length === 0) || (mSelected.length === 0)){
            py_console_log("Must choose one cell and one model from the table.");
            return false;
        }

        var form = document.getElementById("modelpane");
        var batch = document.getElementById("p_batch");
        var cellid = document.getElementById("p_cellid");
        var modelname = document.getElementById("p_modelname");

        batch.value = bSelected;
        cellid.value = cSelected;
        modelname.value = mSelected;

        form.submit();
    });


    ////////////////////////////////////////////////////////////////////
    //////////////////////  PLOTS //////////////////////////////////////
    ////////////////////////////////////////////////////////////////////


    $("#togglePlotOp").on('click',function(){
        var pow = $('#plotOpRow');

        if (pow.css('display') === 'block'){
            pow.css('display', 'none');

        } else if (pow.css('display') === 'none'){
            pow.css('display', 'block');

        } else{
            return false;
        }
    })

    // Default values -- based on 'good' from NarfAnalysis > filter_cells
    if (saved_selections.snr !== null){
        var snr = saved_selections.snr;
    } else{
        var snr = $("#default_snr").val();
    }
    if (saved_selections.iso !== null){
        var iso = saved_selections['iso'];
    } else{
        var iso = $("#default_iso").val();
    }
    if (saved_selections.snri !== null){
        var snri = saved_selections['snri'];
    } else{
        var snri = $("#default_snri").val();
    }

    /*
    var snr = $("#default_snr").val();
    var iso = $("#default_iso").val();
    var snri = $("#default_snri").val();
    */

    $("#plotOpSelect").val('snri');
    $("#plotOpVal").val(snri);

    $("#plotOpSelect").change(updatePlotOpVal);
    function updatePlotOpVal(){
        var select = $("#plotOpSelect");
        var opVal = $("#plotOpVal");
        var getVal = 0.0;

        if (select.val() === 'snr'){
            getVal = snr;
        }
        if (select.val() === 'iso'){
            getVal = iso;
        }
        if (select.val() === 'snri'){
            getVal = snri;
        }
        opVal.val(getVal);
    }

    $("#plotOpVal").change(updateOpVariable);
    function updateOpVariable(){
        var select = $("#plotOpSelect");
        var opVal = $("#plotOpVal");
        var setVal = opVal.val();

        if (select.val() === 'snr'){
            snr = setVal;
            saved_selections.snr = setVal;
        }
        if (select.val() === 'iso'){
            iso = setVal;
            saved_selections.iso = setVal;
        }
        if (select.val() === 'snri'){
            snri = setVal;
            saved_selections.snri = setVal;
        }
    }


    $("#submitPlot").on('click', getNewPlot);
    function getNewPlot(){
        var plotDiv = $("#displayWrapper");

        var plotType = $("#plotTypeSelector").val();
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var measure = $("#measureSelector").val();
        var onlyFair = 0;
        if (document.getElementById("onlyFair").checked){
            onlyFair = 1;
        }
        var includeOutliers = 0;
        if (document.getElementById("includeOutliers").checked){
            includeOutliers = 1;
        }
        var plotNewWindow = 0;

        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/generate_plot_html',
            data: { plotType:plotType, bSelected:bSelected, cSelected:cSelected,
                    mSelected:mSelected, measure:measure, onlyFair:onlyFair,
                    includeOutliers:includeOutliers, iso:iso, snr:snr,
                    snri:snri, plotNewWindow:plotNewWindow },
            type: 'GET',
            success: function(data){
                if (data.hasOwnProperty('script')){
                    if(plotNewWindow){
                        var w = window.open(
                                $SCRIPT_ROOT + '/plot_window',
                                //"_blank",
                                //"width=600, height=600"
                                )
                        $(w.document).ready(function(){
                            w.$(w.document.body).append(data.script + data.div);
                        });
                    } else{
                        $("#statusReportWrapper").html('');
                        plotDiv.html(data.script + data.div);
                    }
                }
                if (data.hasOwnProperty('html')){
                    if(plotNewWindow){
                        var w = window.open(
                                $SCRIPT_ROOT + '/plot_window',
                                //"_blank",
                                //"width=600, height=600"
                                )
                        $(w.document).ready(function(){
                            w.$(w.document.body).append(data.html);
                        });
                    } else{
                        $("#statusReportWrapper").html('');
                        plotDiv.html(data.html);
                    }
                }
                if (data.hasOwnProperty('image')){
                    if(plotNewWindow){
                        var w = window.open(
                            $SCRIPT_ROOT + '/plot_window',
                            )
                        $(w.document).ready(function(){
                            w.$(w.document.body).append(
                                '<img id="preview_image" src="data:image/png;base64,'
                                + data.image + '" />'
                            );
                        });
                    } else{
                        $("#statusReportWrapper").html();
                        plotDiv.html(
                            '<img id="preview_image" src="data:image/png;base64,'
                            + data.image + '" />'
                        );
                    }
                }
                removeLoad();
            },
            error: function(error){
                console.log(error)
                removeLoad();
            }
        });
    }

    $("#submitCustom").on('click', getCustomScript);
    function getCustomScript(){
        var scriptName = $("#customSelector").val();
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var measure = $("#measureSelector").val();
        var onlyFair = 0;
        if (document.getElementById("onlyFair").checked){
            onlyFair = 1;
        }
        var includeOutliers = 0;
        if (document.getElementById("includeOutliers").checked){
            includeOutliers = 1;
        }
        var plotNewWindow = 0;

        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/run_custom',
            data: { scriptName:scriptName, bSelected:bSelected,
                    cSelected:cSelected, mSelected:mSelected, measure:measure,
                    onlyFair:onlyFair, includeOutliers:includeOutliers,
                    iso:iso, snr:snr, snri:snri },
            type: 'GET',
            success: function(data){
                if(odow){
                    var w = window.open(
                        $SCRIPT_ROOT + '/plot_window',
                        //"_blank",
                        //"width=600, height=600"
                        )
                    $(w.document).ready(function(){
                        w.$(w.document.body).append(data.html);
                    });
                } else{
                    $("#statusReportWrapper").html('');
                    $("#displayWrapper").html(data.html);
                    }
                removeLoad();
            },
            error: function(error){
                console.log(error)
                removeLoad();
            }
        });
    }

    $("#reloadScripts").on('click', reloadScripts);
    function reloadScripts(){
        // call scan_for_scripts again in python then,
        // then repopulate the select element with new list
        $.ajax({
            url: $SCRIPT_ROOT + '/reload_scripts',
            data: {},
            type: 'GET',
            success: function(data){
                scripts = $("#customSelector");
                scripts.empty();

                $.each(data.scriptlist, function(sName){
                    scripts.append($("<option></option>")
                        .attr("value", data.scriptlist[sName])
                        .attr("name", "scriptOption[]")
                        .text(data.scriptlist[sName])
                        );
                });
            },
            error: function(error){
                console.log(error)
            }
        })
            var models = $("#modelSelector");
                models.empty();

                $.each(data.modellist, function(modelname) {
                    models.append($("<option></option>")
                        .attr("value", data.modellist[modelname])
                        .attr("name","modelOption[]")
                        .text(data.modellist[modelname]));
                });
    }


    $("#batchPerformance").on('click', batchPerformance);
    function batchPerformance(){
        var bSelected = $("#batchSelector").val();
        var mSelected = $("#modelSelector").val();
        var cSelected = $("#cellSelector").val();
        var findAll = 0;
        if (document.getElementById("findAll").checked){
            findAll = 1;
        }

        var formInfo = document.getElementById('batchPerfForm');
        formInfo.bSelected.value = bSelected;
        formInfo.cSelected.value = cSelected;
        formInfo.mSelected.value = mSelected;
        formInfo.findAll.value = findAll;

        formInfo.submit();
    }

    $("#fitReport").on('click', fitReport);
    function fitReport(){
        var bSelected = $("#batchSelector").val();
        var mSelected = $("#modelSelector").val();
        var cSelected = $("#cellSelector").val();

        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/fit_report',
            data: { bSelected:bSelected, mSelected:mSelected,
                    cSelected:cSelected },
            type: 'GET',
            success: function(data){
                $("#statusReportWrapper").html('');
                $("#displayWrapper").html(
                        '<img id="preview_image" src="data:image/png;base64,'
                        + data.image + '" />'
                        );
                removeLoad();
            },
            error: function(error){
                console.log(error);
                removeLoad();
            }
        });

        // submit as form for new tab
        /*
        var formInfo = document.getElementById('fitRepForm');
        formInfo.bSelected.value = bSelected;
        formInfo.cSelected.value = cSelected;
        formInfo.mSelected.value = mSelected;

        formInfo.submit();
        */
    }


    ///////////////////////////////////////////////////////////////////////////
    ///////////////     Added div toggles / UI management     /////////////////
    ///////////////////////////////////////////////////////////////////////////

    function toggleVisibility(div){
        var sel = div;
        if (sel.css('display') === 'none'){
            sel.css('display', 'block');
        } else if (sel.css('display') === 'block'){
            sel.css('display', 'none');
        } else {
            return false;
        }
    }

    $("#toggleTags").on('click', function(){
        toggleVisibility($("#tagFilters"));
    });

    $("#toggleStatus").on('click', function(){
        toggleVisibility($("#statusFilters"));
    });

    $("#reload").click(function(){
        $.ajax({
            url: $SCRIPT_ROOT + '/reload_modules',
            data: {},
            type: 'GET',
            success: function(data){
                console.log("reload call succeeded");
            },
            error: function(error){
                console.log(error);
            }
        });
    });

    $("#sitemap").click(function(){
        window.open($SCRIPT_ROOT + '/site_map', '_blank');
    });


////////////                 JERB STUFF               ////////////////////

    $("#addJerbKV").click(function(){
        key = $("#jerbKey").val();
        val = $("#jerbVal").val();
        query = $("#jerbQuery").val();
        $.ajax({
            url: $SCRIPT_ROOT + '/add_jerb_kv',
            data: {key:key, val:val, query:query},
            type: 'GET',
            success: function(data){
                $("#jerbQuery").val(JSON.stringify(data.newQuery)).change();
                $("#jerbKey").val('');
                $("#jerbVal").val('');
            },
            error: function(error){
                console.log(error);
            }
        });
    });

    // use the following two functions to enforce exclusion between
    // codeHash and jerbQuery
    /*
    $("#codeHash").change(function(){
        if ($("#codeHash").val() === ''){
            //document.getElementById('jerbQuery').disabled = false;
            return false;
        } else {
            //document.getElementById('jerbQuery').disabled = true;
            $("#jerbQuery").val('');
            $("#jerbKey").val('');
            $("#jerbVal").val('');
        }
    });

    $("#jerbQuery").change(function(){
        if ($("#jerbQuery").val() === ''){
            //document.getElementById('codeHash').disabled = false;
            return false;
        } else {
            //document.getElementById('codeHash').disabled = true;
            $("#codeHash").val('');
        }
    });
    */

    $("#jerbKey").change(function(){
        if ($("#jerbKey").val() === ''){
            return false;
        } else {
            console.log('inside jerbkey function');
            queryString = $("#jerbQuery").val();
            query = JSON.parse(queryString);
            key = $("#jerbKey").val();
            $("#jerbVal").val(query[key]);
        }
    });

    $("#clearQuery").click(function(){
        $("#jerbQuery").val('');
    });
});
