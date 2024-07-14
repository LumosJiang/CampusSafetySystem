package com.example.demo1.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description= "危险信息")
public class Danger {
    @Schema(description = "时间", required = true)
    private String 时间;

    @Schema(description = "地点", required = true)
    private String 地点;

    @Schema(description = "事件shi")
    private String 事件;

    @Schema(description = "图片")
    private  String address;
}
