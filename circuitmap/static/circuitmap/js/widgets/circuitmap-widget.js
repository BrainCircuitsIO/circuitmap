/* -*- mode: espresso; espresso-indent-level: 8; indent-tabs-mode: t -*- */
/* vim: set softtabstop=2 shiftwidth=2 tabstop=2 expandtab: */

(function(CATMAID) {

  "use strict";

  var CircuitmapWidget = function() {
    this.widgetID = this.registerInstance();
    this.idPrefix = `circuitmap-widget${this.widgetID}-`;

    this.fetch_upstream_skeletons = false;
    this.fetch_downstream_skeletons = false;
    this.distance_threshold = 1000;
    this.upstream_syn_count = 5;
    this.downstream_syn_count = 5;
  };

  $.extend(CircuitmapWidget.prototype, new InstanceRegistry());

  CircuitmapWidget.prototype.getName = function() {
    return 'Circuitmap Widget ' + this.widgetID;
  };

  CircuitmapWidget.prototype.getWidgetConfiguration = function() {
    return {
      helpText: 'Circuitmap Widget: ',
      controlsID: this.idPrefix + 'controls',
      createControls: function(controls) {

        var optionFields = document.createElement('div');
        optionFields.innerHTML = `
        <h3>Configuration option for skeleton fetching</h3>
        <table cellpadding="0" cellspacing="0" border="0"
              id="circuitmap_option_fields${this.widgetID}">
            <tr>
              <td><input type="checkbox" name="fetch_upstream_skeletons"
                      id="fetch_upstream_skeletons${this.widgetID}" tabindex="-1" />
                      Fetch upstream autoseg skeletons</td>
              <td></td>
            </tr>
            <tr>
              <td>Minimal synaptic count to fetch upstream skeleton</td>
              <td><input type="number" name="upstream_syn_count" value="5"
                      id="upstream_syn_count${this.widgetID}" tabindex="-1" /></td>
            </tr>
            <tr>
              <td><input type="checkbox" name="fetch_downstream_skeletons"
                    id="fetch_downstream_skeletons${this.widgetID}" tabindex="-1" />
                    Fetch downstream autoseg skeletons</td>
              <td></td>
            </tr>
            <tr>
              <td>Minimal synaptic count to fetch downstream skeleton</td>
              <td><input type="number" name="downstream_syn_count" value="5"
                      id="downstream_syn_count${this.widgetID}" tabindex="-1" /></td>
            </tr>
        </table>
        `;
        controls.appendChild(optionFields);

        var optionFields = document.createElement('div');
        optionFields.innerHTML = `
        <table cellpadding="0" cellspacing="0" border="0"
              id="circuitmap_flow1_option_fields${this.widgetID}">
          <tr>
            <td><h3>Fetch synapses for active skeleton</h3></td>
            <td></td>
          </tr>
          <tr>
            <td>Cut-off distance-from-skeleton threshold (in nm) to retrieve synaptic links</td>
            <td><input type="number" name="distance_threshold" value="1000"
                    id="distance_threshold${this.widgetID}" tabindex="-1" /></td>
          </tr>
        </table>
        <br />
        `;
        controls.appendChild(optionFields);

        var fetch = document.createElement('input');
        fetch.setAttribute("type", "button");
        fetch.setAttribute("value", "Fetch synapses for active neuron");
        fetch.onclick = this.fetch.bind(this);
        controls.appendChild(fetch);

        var optionFields = document.createElement('div');
        optionFields.innerHTML = `
        <table cellpadding="0" cellspacing="0" border="0"
              id="circuitmap_flow2_option_fields${this.widgetID}">
          <tr>
            <td><h3>Fetch autoseg skeleton and synapses at location</h3></td>
            <td></td>
          </tr>
        </table>
        `;
        controls.appendChild(optionFields);

        var fetch = document.createElement('input');
        fetch.setAttribute("type", "button");
        fetch.setAttribute("value", "Fetch autoseg skeleton and synapses at location");
        fetch.onclick = this.fetch_location.bind(this);
        controls.appendChild(fetch);

      },
      contentID: this.idPrefix + 'content',
      createContent: function(container) {
        // container.appendChild(document.createTextNode('Content goes here'));
      },
      init: function() {
        var self = this;

        $('#fetch_upstream_skeleton' + self.widgetID).change(function() {
          self.fetch_upstream_skeleton = this.checked;
        });

        $('#fetch_downstream_skeleton' + self.widgetID).change(function() {
          self.fetch_downstream_skeleton = this.checked;
        });

        $('#distance_threshold' + self.widgetID).change(function() {
          self.distance_threshold = this.value;
        });

        $('#upstream_syn_count' + self.widgetID).change(function() {
          self.upstream_syn_count = this.value;
        });

        $('#downstream_syn_count' + self.widgetID).change(function() {
          self.downstream_syn_count = this.value;
        });

      }

    };
  };

  CircuitmapWidget.prototype.fetch = function() {
    var stackViewer = project.focusedStackViewer;
    var stack = project.focusedStackViewer.primaryStack;

    var query_data = {
      'x': stackViewer.x,
      'y':  stackViewer.y,
      'z':  stackViewer.z,
      'fetch_upstream': this.fetch_upstream_skeletons,
      'fetch_downstream': this.fetch_downstream_skeletons,
      'distance_threshold': this.distance_threshold,
      'upstream_syn_count': this.upstream_syn_count,
      'downstream_syn_count': this.downstream_syn_count,
      'active_skeleton': SkeletonAnnotations.getActiveSkeletonId()
    };

    CATMAID.fetch('ext/circuitmap/' + project.id + '/synapses/fetch', 'POST', query_data)
      .then(function(e) {
        CATMAID.msg("Success", "Import process started ...");
        console.log(e);
    });

  };

  CircuitmapWidget.prototype.fetch_location = function() {
    var stackViewer = project.focusedStackViewer;
    var stack = project.focusedStackViewer.primaryStack;

    var query_data = {
      'x': stackViewer.x,
      'y':  stackViewer.y,
      'z':  stackViewer.z,
      'fetch_upstream': this.fetch_upstream_skeletons,
      'fetch_downstream': this.fetch_downstream_skeletons,
      'distance_threshold': this.distance_threshold,
      'upstream_syn_count': this.upstream_syn_count,
      'downstream_syn_count': this.downstream_syn_count,
      'active_skeleton': -1
    };

    CATMAID.fetch('ext/circuitmap/' + project.id + '/synapses/fetch', 'POST', query_data)
      .then(function(e) {
        CATMAID.msg("Success", "Import process started ...");
        console.log(e);
    });

  };

  CircuitmapWidget.prototype.destroy = function() {
    this.unregisterInstance();
  };

  CATMAID.registerWidget({
    name: 'Circuit Map Widget',
    description: 'Widget associated with the circuitmap extension',
    key: 'circuitmap-widget',
    creator: CircuitmapWidget
  });

})(CATMAID);
