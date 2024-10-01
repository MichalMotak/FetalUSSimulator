using UnityEngine;
using UnityEngine.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
using System.Text;
using System.IO;
using System.Threading.Tasks;

namespace UnityVolumeRendering
{
	public class SavePlane: MonoBehaviour
	{		
		private List<string[]> rowData = new List<string[]>();
		private string filePath = "AcquiredData/Poses1/labels.csv";
		private string imagePath = "AcquiredData";
		
		private float pos_x = -0.01f, pos_y = -0.05f, pos_z = 0.0f;
		private float rot_x = -150f, rot_y = -100f, rot_z = 70.8f;

		// private float y_offset_up = 21.0f;
		private float y_offset_up = 500;

		private float y_offset_down = 100.0f;

		private float x_offset_left = 500.0f;

		private int waitForMilliSeconds = 1000;

		// Standard tests

		public int screenshotIndex = 0;

	    // Start is called before the first frame update
	    void Start()
	    {
	    	// Creating First row of titles manually
	    	string[] rowDataTemp = new string[7];
			rowDataTemp[0] = "id";
	        rowDataTemp[1] = "pos_x";
	        rowDataTemp[2] = "pos_y";
	        rowDataTemp[3] = "pos_z";
	        rowDataTemp[4] = "rot_x";
	        rowDataTemp[5] = "rot_y";
	        rowDataTemp[6] = "rot_z";
	        rowData.Add(rowDataTemp);

	        EditorApplication.ExecuteMenuItem("Volume Rendering/Slice acquisition");

			SetPose();
	    }

	    public async Task MyAsyncMethod()
		{
		    await Task.Delay(waitForMilliSeconds);
		}

	    // Update is called once per frame
	    public async void SetPose()
	    {
			Debug.Log("SAVE PLANE ");

			var activeWindow = EditorWindow.focusedWindow;
			Debug.Log("active win " + activeWindow);
			var vec2Position = activeWindow.position.position;
			var sizeX = activeWindow.position.width;
			var sizeY = activeWindow.position.height;
			var sizeX_plane = sizeX; // - sizeX * 0.5f;
			var sizeY_plane = sizeY - y_offset_down;

			Vector3 rot = new Vector3(rot_x, rot_y, rot_z);
			transform.localRotation = Quaternion.Euler(rot);

			Vector3 pos = new Vector3(pos_x, pos_y, pos_z);
			transform.localPosition = pos;

			string img_name = "plane" + (screenshotIndex+0) + ".png";

			await MyAsyncMethod();

				//Debug.Log(vec2Position);
				//Vector2 vec = new Vector2((vec2Position.x + x_offset_left), (vec2Position.y + y_offset_up));
				//Debug.Log("vec2Pos");
				Debug.Log($"vec2Position {vec2Position}");
				//Debug.Log($"vec {vec}");
				Debug.Log($"sizeX_plane {sizeX_plane}");
				Debug.Log($"sizeY_plane {sizeY_plane}");
				Vector2 vec = new Vector2(100,100);
				Debug.Log($"vec {vec}");



				var colors = InternalEditorUtility.ReadScreenPixel(vec, (int)sizeX_plane, (int)sizeY_plane);
				var result = new Texture2D((int)sizeX_plane, (int)sizeY_plane, TextureFormat.RGB24, false);
				result.SetPixels(colors);
				var bytes = result.EncodeToPNG();
				DestroyImmediate(result);
				File.WriteAllBytes(Path.Combine(imagePath, "plane" + screenshotIndex + ".png"), bytes);
				Debug.Log("screenshotIndex "+ screenshotIndex);
				screenshotIndex++;
				AssetDatabase.Refresh();

				//ScreenCapture.CaptureScreenshot(Path.Combine(imagePath, "planeeee" + screenshotIndex + ".png"));
							
		}
	}
}

#endif

