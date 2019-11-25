/* -*- mode: espresso; espresso-indent-level: 8; indent-tabs-mode: t -*- */
/* vim: set softtabstop=2 shiftwidth=2 tabstop=2 expandtab: */

(function(CATMAID) {

  "use strict";

  var CircuitmapWidget = function() {
    this.widgetID = this.registerInstance();
    this.idPrefix = `circuitmap-widget${this.widgetID}-`;

    this.fetch_upstream_skeletons = false;
    this.fetch_downstream_skeletons = false;
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
        <table cellpadding="0" cellspacing="0" border="0"
              id="circuitmap_option_fields${this.widgetID}">
          <tr>
            <td><input type="checkbox" name="fetch_upstream_skeletons"
                    id="fetch_upstream_skeletons${this.widgetID}" tabindex="-1" /></td>
            <td>Fetch upstream skeletons</td>
          </tr>
          <tr>
            <td><input type="checkbox" name="fetch_downstream_skeletons"
                    id="fetch_downstream_skeletons${this.widgetID}" tabindex="-1" /></td>
            <td>Fetch downstream skeletons</td>
          </tr>
        </table>
        `;
        controls.appendChild(optionFields);

        var fetch = document.createElement('input');
        fetch.setAttribute("type", "button");
        fetch.setAttribute("value", "Fetch");
        fetch.onclick = this.fetch.bind(this);
        controls.appendChild(fetch);


      },
      contentID: this.idPrefix + 'content',
      createContent: function(container) {
        container.appendChild(document.createTextNode('Content goes here'));
      },
      init: function() {
        var self = this;

        $('#fetch_upstream_skeleton' + self.widgetID).change(function() {
          self.fetch_upstream_skeleton = this.checked;
        });

      }

    };
  };

  CircuitmapWidget.prototype.fetch = function() {
    var stackViewer = project.focusedStackViewer;
    console.log('coordinates are ...', stackViewer.x, stackViewer.y, stackViewer.z);
    console.log('fetch upstream?', this.fetch_upstream_skeletons);
    console.log('fetch downstream?', this.fetch_downstream_skeletons);
    console.log('current project id', project.id);
    console.log('skeleton id selected?', SkeletonAnnotations.getActiveSkeletonId() );
    /*$.ajax({
      'url': 'https://cloud.braincircuits.io/api/v1/segment/lookup/' + activeStackViewer.x + '/' + activeStackViewer.y + '/' + activeStackViewer.z,
      'type': 'GET'
    });*/
  };

  CircuitmapWidget.prototype.destroy = function() {
    this.unregisterInstance();
  };

  CATMAID.registerWidget({
    name: 'Circuit Map Widget',
    description: 'Widget associated with the circuitmap app',
    key: 'circuitmap-widget',
    creator: CircuitmapWidget
  });

})(CATMAID);
