package com.example.demo1.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description= "视频信息")
public class Video {
    @Schema(description = "视频id", required = true)
    private int video_id;

    @Schema(description = "视频路径", required = true)
    private String path;

    @Schema(description = "视频日期")
    private String date;

    @Schema(description = "视频名")
    private String name;

    @Schema(description = "类型")
    private String type;
}
